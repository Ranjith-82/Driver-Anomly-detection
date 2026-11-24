from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import threading
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from evaluation.common.inputs import (
    EvaluationInputError,
    GAZE_LABELS,
    label_at,
    load_ground_truth,
    normalize_gaze_label,
    normalize_binary_label,
)
from evaluation.common.evaluators import evaluate_fatigue_events
from evaluation.common.metrics import classification_metrics, detection_metrics
from evaluation.common.reports import (
    write_confusion_matrix_png,
    write_json,
    write_metrics_csv,
    write_predictions_csv,
)
from evaluation.common.video import iter_video_frames


BEHAVIOR_CLASSES = ("NORMAL", "YAWNING", "DISTRACTED", "TALKING")
GAZE_DIRECTIONAL_CLASSES = tuple(label for label in GAZE_LABELS if label != "UNKNOWN")
FATIGUE_CONFIRMATION_SECONDS = 1.5
GAZE_DISTRACTION_PERSISTENCE_SECONDS = 0.6


class _ClassifierRecorder:
    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier
        self.last_probabilities: list[float] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.classifier, name)

    def predict_proba(self, features: Any) -> Any:
        probabilities = self.classifier.predict_proba(features)
        self.last_probabilities = [float(value) for value in probabilities[0]]
        return probabilities


def _optional_import_error(package: str, extra: str) -> RuntimeError:
    return RuntimeError(
        f"{package} is required for this evaluation; install project extra {extra!r}."
    )


def _load_gaze_calibration(repository_root: Path, calibration_path: Path):
    from driver_monitor.legacy import load_legacy_module

    calibration_module_path = (
        repository_root
        / "Phase-2"
        / "Gaze_detection"
        / "gaze_detection.py"
    )
    calibration_module = load_legacy_module(
        "gaze_calibration",
        calibration_module_path,
    )
    return calibration_module.load_calibration(calibration_path)


def _load_behavior_pipeline(calibration_path: Path):
    try:
        import driver_monitor.monitoring.legacy_runtime as runtime
        from driver_monitor.config import ProjectPaths
        from driver_monitor.detection import behavior, gaze
        from driver_monitor.monitoring.temporal import BehaviorTemporalFilter
        from driver_monitor.models import FramePacket
    except ImportError as error:
        raise _optional_import_error("the behavior runtime", "monitor") from error

    paths = ProjectPaths.discover()
    for required in (paths.behavior_model, paths.behavior_labels, paths.face_landmarker):
        if not required.is_file():
            raise FileNotFoundError(f"Required project model is missing: {required}")
    observations = _load_gaze_calibration(
        paths.repository_root,
        calibration_path,
    )
    gaze_thresholds = gaze.calibrate_gaze_thresholds(observations)

    with paths.behavior_model.open("rb") as file:
        classifier = pickle.load(file)
    # Offline evaluation predicts one frame at a time; avoid all-core pool overhead.
    classifier.n_jobs = 1
    classifier = _ClassifierRecorder(classifier)
    with paths.behavior_labels.open("rb") as file:
        encoder = pickle.load(file)

    result_buffer = runtime.ResultBuffer()
    worker = runtime.BehaviorWorker(
        None,
        result_buffer,
        threading.Event(),
        paths.behavior_model,
        paths.behavior_labels,
        paths.face_landmarker,
        0.65,
        25.0,
        20.0,
        0.62,
        0.21,
        0.24,
        FATIGUE_CONFIRMATION_SECONDS,
        GAZE_DISTRACTION_PERSISTENCE_SECONDS,
    )
    worker.temporal_filter = BehaviorTemporalFilter(
        talking_seconds=0.35,
        yawning_seconds=0.15,
        distraction_seconds=0.1,
        clear_seconds=0.15,
    )
    worker.gaze_thresholds = gaze_thresholds

    def reset_video_state():
        worker.temporal_filter = BehaviorTemporalFilter(
            talking_seconds=0.35,
            yawning_seconds=0.15,
            distraction_seconds=0.1,
            clear_seconds=0.15,
        )
        result_buffer.update(runtime.ResultBuffer().snapshot())
        detector = behavior.create_landmark_detector(paths.face_landmarker)
        pose = behavior.create_head_pose_estimator()
        mouth = behavior.create_mouth_analyzer(mar_threshold=0.62)
        eyes = behavior.create_eye_analyzer(
            close_threshold=0.21,
            open_threshold=0.24,
            fatigue_seconds=FATIGUE_CONFIRMATION_SECONDS,
        )
        state = behavior.create_behavior_state_machine(
            yaw_threshold=25.0,
            pitch_threshold=20.0,
        )
        history = deque(maxlen=9)
        smoother = gaze.create_gaze_smoother()
        distraction_timer = gaze.create_distraction_timer(
            GAZE_DISTRACTION_PERSISTENCE_SECONDS
        )
        return detector, pose, mouth, eyes, state, history, smoother, distraction_timer

    return (
        runtime,
        worker,
        result_buffer,
        classifier,
        encoder,
        FramePacket,
        reset_video_state,
    )


def _behavior_rows(
    video_path: Path,
    calibration_path: Path,
    task: str,
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    (
        _,
        worker,
        result_buffer,
        classifier,
        encoder,
        frame_packet,
        reset_video_state,
    ) = _load_behavior_pipeline(calibration_path)
    detector, pose, mouth, eyes, state, history, smoother, distraction_timer = (
        reset_video_state()
    )
    rows: list[dict[str, Any]] = []
    try:
        for frame_index, timestamp, frame in iter_video_frames(video_path):
            packet = frame_packet(frame_index, timestamp, frame)
            classifier.last_probabilities = None
            worker.process_packet(
                packet,
                detector,
                pose,
                mouth,
                eyes,
                state,
                classifier,
                encoder,
                history,
                smoother,
                distraction_timer,
            )
            result = result_buffer.snapshot()
            actual_source = label_at(intervals, video_path.name, timestamp)
            actual = actual_source
            if task == "fatigue":
                predicted = "positive" if result["fatigue_alert"] else "negative"
                actual = (
                    normalize_binary_label(actual_source)
                    if actual_source is not None else None
                )
            elif task == "gaze":
                predicted = str(result["gaze_direction"])
                actual = (
                    normalize_gaze_label(actual_source)
                    if actual_source is not None else None
                )
            else:
                predicted = str(result["behavior"])
                actual = actual.upper() if actual is not None else None
            predicted_label = str(predicted).strip().upper()
            raw_gaze_label = "UNKNOWN"
            if (
                task == "gaze"
                and result.get("gaze_x") is not None
                and result.get("gaze_y") is not None
            ):
                from driver_monitor.detection import gaze

                raw_gaze_label = gaze.classify_gaze(
                    SimpleNamespace(
                        x=float(result["gaze_x"]),
                        y=float(result["gaze_y"]),
                    ),
                    worker.gaze_thresholds,
                )
            raw_model_behavior = ""
            model_confidence: float | str = ""
            if classifier.last_probabilities:
                index = max(
                    range(len(classifier.last_probabilities)),
                    key=classifier.last_probabilities.__getitem__,
                )
                raw_model_behavior = str(
                    encoder.inverse_transform([index])[0]
                ).strip().upper()
                model_confidence = classifier.last_probabilities[index]
            calibration_status = (
                "CALIBRATING"
                if str(result.get("raw_behavior", "")).strip().upper()
                == "CALIBRATING"
                else "READY"
            )
            row = {
                "video_name": video_path.name,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "actual_label": actual.strip().upper() if actual else "",
                "predicted_label": predicted_label,
                "confidence": result.get("behavior_confidence", ""),
                "face_present": result.get("face_present", False),
                "raw_behavior": str(result.get("raw_behavior", "")).strip().upper(),
                "raw_model_behavior": raw_model_behavior,
                "model_confidence": model_confidence,
                "raw_gaze_label": raw_gaze_label,
                "calibration_status": calibration_status,
                "prediction_valid": (
                    bool(result.get("face_present"))
                    and (
                        task != "gaze"
                        or predicted_label in GAZE_DIRECTIONAL_CLASSES
                    )
                ),
                "fatigue_alert": result.get("fatigue_alert", False),
                "eyes_closed_duration": result.get("eyes_closed_duration", 0.0),
                "eyes_state": result.get("eyes_state", "EYE STATE UNCERTAIN"),
                "average_ear": result.get("average_ear"),
                "gaze_direction": result.get("gaze_direction", ""),
                "gaze_x": result.get("gaze_x"),
                "gaze_y": result.get("gaze_y"),
                "gaze_distraction_elapsed": result.get("gaze_distraction_elapsed", 0.0),
                "gaze_distraction_confirmed": result.get(
                    "gaze_distraction_confirmed", False
                ),
            }
            if task == "behavior":
                row["evaluation_status"] = _behavior_evaluation_status(row)
            elif task == "gaze":
                row["evaluation_status"] = _gaze_evaluation_status(row)
            rows.append(row)
    finally:
        detector.close()
    return rows


def _behavior_evaluation_status(row: Mapping[str, Any]) -> str:
    if not str(row.get("actual_label", "")).strip():
        return "UNLABELLED"
    if (
        str(row.get("calibration_status", "")).strip().upper() == "CALIBRATING"
        or str(row.get("predicted_label", "")).strip().upper() == "CALIBRATING"
    ):
        return "EXCLUDED_CALIBRATING"
    if (
        str(row.get("face_present", "true")).strip().casefold() == "false"
        or str(row.get("predicted_label", "")).strip().upper()
        == "FACE NOT DETECTED"
    ):
        return "EXCLUDED_FACE_NOT_DETECTED"
    if str(row.get("predicted_label", "")).strip().upper() not in BEHAVIOR_CLASSES:
        return "EXCLUDED_UNCLASSIFIED"
    return "EVALUATED"


def _gaze_evaluation_status(row: Mapping[str, Any]) -> str:
    if not str(row.get("actual_label", "")).strip():
        return "UNLABELLED"
    if str(row.get("calibration_status", "")).strip().upper() == "CALIBRATING":
        return "EXCLUDED_CALIBRATING"
    if str(row.get("actual_label", "")).strip().upper() == "UNKNOWN":
        return "EXCLUDED_UNKNOWN_GROUND_TRUTH"
    if (
        str(row.get("face_present", "true")).strip().casefold() == "false"
        or str(row.get("predicted_label", "")).strip().upper()
        not in GAZE_DIRECTIONAL_CLASSES
    ):
        return "EXCLUDED_INVALID_PREDICTION"
    return "EVALUATED"


def _recognition_rows(
    video_path: Path,
    database_path: Path,
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
    tolerance: float,
    resize_scale: float,
) -> list[dict[str, Any]]:
    try:
        from driver_monitor.detection.recognition import (
            create_live_recognizer,
            recognize_frame,
        )
        from driver_monitor.profiles.repository import DriverProfileRepository
    except ImportError as error:
        raise _optional_import_error("face_recognition", "recognition") from error

    if not database_path.is_file():
        raise FileNotFoundError(f"Driver database does not exist: {database_path}")
    recognizer = create_live_recognizer(DriverProfileRepository(database_path))
    if not recognizer.known_encodings:
        raise ValueError(
            f"No enrolled driver face encodings could be loaded from {database_path}"
        )
    rows: list[dict[str, Any]] = []
    for frame_index, timestamp, frame in iter_video_frames(video_path):
        predictions = recognize_frame(
            frame,
            recognizer,
            tolerance=tolerance,
            resize_scale=resize_scale,
        )
        if len(predictions) > 1:
            raise EvaluationInputError(
                f"{video_path.name} frame {frame_index} contains multiple faces; "
                "identity intervals require one driver face per frame."
            )
        identity, confidence, location = predictions[0] if predictions else (None, 0.0, None)
        rows.append({
            "video_name": video_path.name,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "actual_label": (
                "unknown"
                if (actual := label_at(intervals, video_path.name, timestamp)) is not None
                and actual.casefold() == "unknown"
                else actual or ""
            ),
            "predicted_label": str(identity) if identity is not None else "unknown",
            "confidence": confidence,
            "face_location": json.dumps(location) if location is not None else "",
        })
    return rows


def _emotion_rows(
    video_path: Path,
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    try:
        from driver_monitor.config import ProjectPaths
        from driver_monitor.detection import behavior
        from driver_monitor.detection.svm_emotion import SVMEmotionAnalyzer
        from driver_monitor.monitoring.workers import SVMEmotionWorker
    except ImportError as error:
        raise _optional_import_error("the SVM emotion runtime", "monitor") from error

    paths = ProjectPaths.discover()
    if not paths.face_landmarker.is_file():
        raise FileNotFoundError(f"Face-landmarker model is missing: {paths.face_landmarker}")
    detector = behavior.create_landmark_detector(paths.face_landmarker)
    rows: list[dict[str, Any]] = []
    try:
        analyzer = SVMEmotionAnalyzer(paths.emotion_model_dir)
        try:
            for frame_index, timestamp, frame in iter_video_frames(video_path):
                landmarks = detector.extract(frame)
                prediction = "unknown"
                if landmarks is not None:
                    frame_height, frame_width = frame.shape[:2]
                    bounds = (
                        max(0, int(landmarks[:, 1].min() * frame_height)),
                        min(frame_width, int(landmarks[:, 0].max() * frame_width)),
                        min(frame_height, int(landmarks[:, 1].max() * frame_height)),
                        max(0, int(landmarks[:, 0].min() * frame_width)),
                    )
                    crop = SVMEmotionWorker._crop_face(frame, bounds)
                    if crop is not None and crop.size:
                        prediction = analyzer.predict(crop) or "unknown"
                actual = label_at(intervals, video_path.name, timestamp)
                rows.append({
                    "video_name": video_path.name,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "actual_label": actual.casefold() if actual is not None else "",
                    "predicted_label": prediction,
                })
        finally:
            analyzer.close()
    finally:
        detector.close()
    return rows


def _load_yolo(model_path: Path):
    try:
        from driver_monitor.detection.objects import YoloObjectDetector
    except ImportError as error:
        raise _optional_import_error("Ultralytics YOLO", "monitor") from error
    return YoloObjectDetector(model_path)


def _load_box_annotations(
    path: Path,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], set[tuple[str, int]]]:
    required = {
        "video_name", "frame_index", "class_name",
        "x_min", "y_min", "x_max", "y_max",
    }
    annotations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    annotated_frames: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        import csv

        reader = csv.DictReader(file)
        fieldnames = {
            field.strip() for field in (reader.fieldnames or ()) if field is not None
        }
        missing = required - fieldnames
        if missing:
            if {
                "video_name", "start_time", "end_time", "actual_label"
            }.issubset(fieldnames):
                raise EvaluationInputError(
                    "This CSV contains video-level label intervals, not bounding "
                    "boxes. YOLO mAP needs one row per labelled frame/object with "
                    "video_name,frame_index,class_name,x_min,y_min,x_max,y_max; "
                    "include a blank class_name and coordinates for every "
                    "annotated frame with no objects."
                )
            raise EvaluationInputError(
                f"Bounding-box CSV is missing columns: {sorted(missing)}. "
                "Required header: "
                "video_name,frame_index,class_name,x_min,y_min,x_max,y_max."
            )
        for line_number, row in enumerate(reader, start=2):
            video_name = (row["video_name"] or "").strip()
            if not video_name:
                raise EvaluationInputError(f"Missing video_name on box row {line_number}")
            try:
                frame_index = int(row["frame_index"])
            except ValueError as error:
                raise EvaluationInputError(
                    f"Invalid frame_index on box row {line_number}"
                ) from error
            if frame_index < 0:
                raise EvaluationInputError(f"Negative frame_index on row {line_number}")
            key = video_name, frame_index
            annotated_frames.add(key)
            class_name = (row["class_name"] or "").strip()
            if not class_name:
                coordinates = [
                    (row[column] or "").strip()
                    for column in ("x_min", "y_min", "x_max", "y_max")
                ]
                if any(coordinates):
                    raise EvaluationInputError(
                        f"Empty class_name has box coordinates on row {line_number}"
                    )
                continue
            try:
                box = [float(row[column]) for column in ("x_min", "y_min", "x_max", "y_max")]
                class_id = int(class_name) if class_name.isdigit() else None
            except ValueError as error:
                raise EvaluationInputError(
                    f"Invalid bounding box on row {line_number}"
                ) from error
            if (
                not all(math.isfinite(value) and value >= 0 for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise EvaluationInputError(
                    f"Invalid xyxy box coordinates on row {line_number}"
                )
            annotations[key].append({
                "class_id": class_id,
                "class_name": class_name,
                "box": box,
            })
    return dict(annotations), annotated_frames


def _yolo_rows(
    videos: Sequence[Path],
    model_path: Path,
    annotations_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from driver_monitor.monitoring.legacy_runtime import (
        OBJECT_DETECTION_CLASS_CONFIDENCE,
        OBJECT_DETECTION_IMAGE_SIZE,
        OBJECT_DETECTION_IOU_THRESHOLD,
        OBJECT_DETECTION_MAX_DETECTIONS,
    )
    from driver_monitor.detection.objects import OBJECT_CLASSES

    annotations, annotated_frames = _load_box_annotations(annotations_path)
    expected_frames: set[tuple[str, int]] = set()
    for video_path in videos:
        for frame_index, _, _ in iter_video_frames(video_path):
            key = video_path.name, frame_index
            expected_frames.add(key)
    missing_frames = sorted(expected_frames - annotated_frames)
    if missing_frames:
        raise EvaluationInputError(
            f"YOLO annotations are incomplete: {len(missing_frames)} decoded "
            f"frames have no annotation row. First missing frame: "
            f"{missing_frames[0][0]} frame {missing_frames[0][1]}."
        )
    unexpected_frames = sorted(annotated_frames - expected_frames)
    if unexpected_frames:
        raise EvaluationInputError(
            f"YOLO annotations include {len(unexpected_frames)} frame(s) not "
            f"present in the selected videos. First extra frame: "
            f"{unexpected_frames[0][0]} frame {unexpected_frames[0][1]}."
        )

    detector = _load_yolo(model_path)
    prediction_rows: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    filtered_samples: list[dict[str, Any]] = []
    valid_class_names = {name.casefold(): class_id for class_id, name in OBJECT_CLASSES.items()}
    for video_path in videos:
        for frame_index, timestamp, frame in iter_video_frames(video_path):
            key = video_path.name, frame_index
            raw_detections = detector.detect(
                frame,
                image_size=OBJECT_DETECTION_IMAGE_SIZE,
                confidence=0.001,
                iou=OBJECT_DETECTION_IOU_THRESHOLD,
                max_detections=OBJECT_DETECTION_MAX_DETECTIONS,
                class_confidence=None,
                max_box_fraction=1.0,
            )
            truths = []
            for annotation in annotations.get(key, []):
                class_id = annotation["class_id"]
                if class_id is None:
                    class_id = valid_class_names.get(annotation["class_name"].casefold())
                if class_id not in OBJECT_CLASSES:
                    raise EvaluationInputError(
                        f"Unknown YOLO class {annotation['class_name']!r} "
                        f"for {video_path.name} frame {frame_index}"
                    )
                _, _, x2, y2 = annotation["box"]
                if x2 > frame.shape[1] or y2 > frame.shape[0]:
                    raise EvaluationInputError(
                        f"YOLO annotation exceeds frame bounds for "
                        f"{video_path.name} frame {frame_index}"
                    )
                truths.append({"class_id": class_id, "box": annotation["box"]})
            raw_predictions = [
                {
                    "class_id": detection["class_id"],
                    "box": detection["box"],
                    "confidence": detection["confidence"],
                }
                for detection in raw_detections
            ]
            filtered_predictions = [
                prediction for prediction in raw_predictions
                if prediction["confidence"]
                >= OBJECT_DETECTION_CLASS_CONFIDENCE.get(
                    prediction["class_id"], 1.0
                )
                and (
                    max(0, prediction["box"][2] - prediction["box"][0])
                    * max(0, prediction["box"][3] - prediction["box"][1])
                    <= 0.85 * frame.shape[0] * frame.shape[1]
                )
            ]
            raw_samples.append({
                "video_name": video_path.name,
                "frame_index": frame_index,
                "ground_truth": truths,
                "predictions": raw_predictions,
            })
            filtered_samples.append({
                "video_name": video_path.name,
                "frame_index": frame_index,
                "ground_truth": truths,
                "predictions": filtered_predictions,
            })
            for stage, predictions in (
                ("raw_model", raw_predictions),
                ("confidence_filtered", filtered_predictions),
            ):
                if predictions:
                    for prediction in predictions:
                        prediction_rows.append({
                            "video_name": video_path.name,
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "prediction_stage": stage,
                            "predicted_class": OBJECT_CLASSES[prediction["class_id"]],
                            "confidence": prediction["confidence"],
                            "x_min": prediction["box"][0],
                            "y_min": prediction["box"][1],
                            "x_max": prediction["box"][2],
                            "y_max": prediction["box"][3],
                        })
                else:
                    prediction_rows.append({
                        "video_name": video_path.name,
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "prediction_stage": stage,
                        "predicted_class": "",
                        "confidence": "",
                        "x_min": "",
                        "y_min": "",
                        "x_max": "",
                        "y_max": "",
                    })
    filtered_metrics = detection_metrics(filtered_samples, OBJECT_CLASSES)
    raw_metrics = detection_metrics(raw_samples, OBJECT_CLASSES)
    filtered_metrics["raw_model_metrics"] = raw_metrics
    filtered_metrics["confidence_filter"] = {
        OBJECT_CLASSES[class_id]: threshold
        for class_id, threshold in OBJECT_DETECTION_CLASS_CONFIDENCE.items()
    }
    filtered_metrics["final_alert_metrics_status"] = (
        "unavailable: live phone persistence/proximity rules require an "
        "independent driver identity/face-location annotation stream"
    )
    return prediction_rows, filtered_metrics


def _evaluate_classification(
    rows: Sequence[Mapping[str, Any]],
    task: str,
) -> dict[str, Any]:
    labelled_rows = [
        row for row in rows
        if row.get("actual_label") not in (None, "")
    ]
    if task == "fatigue":
        calibration = [
            row for row in labelled_rows
            if str(row.get("calibration_status", "")).upper() == "CALIBRATING"
        ]
        face_missing = [
            row for row in labelled_rows
            if not _face_present(row.get("face_present", True), 0)
        ]
        invalid_eye_state = [
            row for row in labelled_rows
            if (
                row not in calibration
                and row not in face_missing
                and (
                    row.get("average_ear") in (None, "")
                    or str(row.get("eyes_state", "")).strip().upper()
                    == "EYE STATE UNCERTAIN"
                )
            )
        ]
        valid_rows = [
            row for row in labelled_rows
            if (
                row not in calibration
                and row not in face_missing
                and row not in invalid_eye_state
            )
        ]
        actual = [normalize_binary_label(str(row["actual_label"])) for row in valid_rows]
        predicted = [str(row["predicted_label"]).strip().casefold() for row in valid_rows]
        if any(label not in {"positive", "negative"} for label in predicted):
            raise EvaluationInputError("Fatigue predictions must be positive or negative.")
        metrics = classification_metrics(actual, predicted, ("negative", "positive"))
        eye_closure_predictions = [
            "positive"
            if str(row.get("eyes_state", "")).strip().upper() == "EYES CLOSED"
            else "negative"
            for row in valid_rows
        ]
        sustained_closure_predictions = [
            "positive"
            if float(row.get("eyes_closed_duration", 0.0))
            >= FATIGUE_CONFIRMATION_SECONDS
            else "negative"
            for row in valid_rows
        ]
        metrics.update({
            "total_processed_frames": len(rows),
            "labelled_frames": len(labelled_rows),
            "unlabelled_frames_excluded": len(rows) - len(labelled_rows),
            "calibration_frames_excluded": len(calibration),
            "face_not_detected_frames_excluded": len(face_missing),
            "invalid_eye_state_frames_excluded": len(invalid_eye_state),
            "evaluated_frames": len(valid_rows),
            "eye_closure_confirmation_threshold_seconds": (
                FATIGUE_CONFIRMATION_SECONDS
            ),
            "decision_stages": {
                "eye_closure_detection": classification_metrics(
                    actual, eye_closure_predictions, ("negative", "positive")
                ),
                "sustained_eye_closure": classification_metrics(
                    actual, sustained_closure_predictions, ("negative", "positive")
                ),
                "final_fatigue_alert": dict(metrics),
            },
        })
        return metrics
    valid_rows = labelled_rows
    actual = [str(row["actual_label"]).strip().casefold() for row in valid_rows]
    predicted = [str(row["predicted_label"]).strip().casefold() for row in valid_rows]
    classes = list(dict.fromkeys([*actual, *predicted])) or ["unknown"]
    result = classification_metrics(actual, predicted, classes)
    result.update({
        "total_processed_frames": len(rows),
        "labelled_frames": len(labelled_rows),
        "unlabelled_frames_excluded": len(rows) - len(labelled_rows),
        "evaluated_frames": len(valid_rows),
    })
    return result


def _face_present(value: Any, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise EvaluationInputError(
        f"Behavior prediction row {row_number} has invalid face_present value "
        f"{value!r}."
    )


def _behavior_classification_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the classifier output and final decision on valid face samples."""
    labelled_rows = [
        row for row in rows
        if str(row.get("actual_label", "")).strip()
    ]
    unlabelled_frames = len(rows) - len(labelled_rows)
    calibration_rows = [
        row for row in labelled_rows
        if (
            str(row.get("calibration_status", "")).strip().upper() == "CALIBRATING"
            or str(row.get("predicted_label", "")).strip().upper() == "CALIBRATING"
        )
    ]
    post_calibration_rows = [row for row in labelled_rows if row not in calibration_rows]
    face_missing_rows: list[Mapping[str, Any]] = []
    unclassified_rows: list[Mapping[str, Any]] = []
    final_rows: list[Mapping[str, Any]] = []
    for row_number, row in enumerate(post_calibration_rows, start=1):
        actual = str(row["actual_label"]).strip().upper()
        if actual not in BEHAVIOR_CLASSES:
            raise EvaluationInputError(
                f"Behavior prediction row {row_number} has unsupported ground-truth "
                f"label {actual!r}; expected one of {BEHAVIOR_CLASSES}."
            )
        predicted = str(row.get("predicted_label", "")).strip().upper()
        face_present = _face_present(row.get("face_present"), row_number)
        if not face_present or predicted == "FACE NOT DETECTED":
            face_missing_rows.append(row)
        elif predicted not in BEHAVIOR_CLASSES:
            unclassified_rows.append(row)
        else:
            final_rows.append(row)

    final_metrics = classification_metrics(
        [str(row["actual_label"]).strip().upper() for row in final_rows],
        [str(row["predicted_label"]).strip().upper() for row in final_rows],
        BEHAVIOR_CLASSES,
    )
    raw_model_rows = [
        row for row in post_calibration_rows
        if _face_present(row.get("face_present"), 0)
        and str(row.get("raw_model_behavior", "")).strip().upper()
        in BEHAVIOR_CLASSES
    ]
    raw_model_metrics = classification_metrics(
        [str(row["actual_label"]).strip().upper() for row in raw_model_rows],
        [str(row["raw_model_behavior"]).strip().upper() for row in raw_model_rows],
        BEHAVIOR_CLASSES,
    )
    face_detection_failures = sum(
        not _face_present(row.get("face_present"), row_number)
        for row_number, row in enumerate(labelled_rows, start=1)
    )
    macro_fpr = sum(
        float(item["false_positive_rate"])
        for item in final_metrics["per_class"].values()
    ) / len(BEHAVIOR_CLASSES)
    macro_fnr = sum(
        float(item["false_negative_rate"])
        for item in final_metrics["per_class"].values()
    ) / len(BEHAVIOR_CLASSES)
    raw_final_disagreements = sum(
        str(row.get("raw_behavior", "")).strip().upper()
        != str(row.get("predicted_label", "")).strip().upper()
        for row in post_calibration_rows
        if str(row.get("raw_behavior", "")).strip()
    )
    return {
        **final_metrics,
        "precision": final_metrics["macro_precision"],
        "recall": final_metrics["macro_recall"],
        "macro_false_positive_rate": macro_fpr,
        "macro_false_negative_rate": macro_fnr,
        "classes": list(BEHAVIOR_CLASSES),
        "raw_model_metrics": raw_model_metrics,
        "total_processed_frames": len(rows),
        "labelled_frames": len(labelled_rows),
        "unlabelled_frames_excluded": unlabelled_frames,
        "calibration_frames_processed": sum(
            str(row.get("calibration_status", "")).strip().upper() == "CALIBRATING"
            or str(row.get("predicted_label", "")).strip().upper() == "CALIBRATING"
            for row in rows
        ),
        "calibration_frames_excluded": len(calibration_rows),
        "face_not_detected_excluded_frames": len(face_missing_rows),
        "unclassified_excluded_frames": len(unclassified_rows),
        "excluded_frames": (
            unlabelled_frames + len(calibration_rows)
            + len(face_missing_rows) + len(unclassified_rows)
        ),
        "evaluated_frames": final_metrics["samples"],
        "face_detection_expected_frames": len(labelled_rows),
        "face_detection_failures": face_detection_failures,
        "face_detection_failure_rate": (
            face_detection_failures / len(labelled_rows) if labelled_rows else 0.0
        ),
        "face_detection_rate": (
            (len(labelled_rows) - face_detection_failures) / len(labelled_rows)
            if labelled_rows else 0.0
        ),
        "face_not_detected_output_frames": len(face_missing_rows),
        "raw_vs_final_disagreement_frames": raw_final_disagreements,
        "prediction_source": (
            "Four-class confusion matrix excludes calibration, face-missing, and "
            "unclassified outputs; counts for each are reported separately."
        ),
    }


def _gaze_classification_metrics(
    rows: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    labelled_rows = [row for row in rows if str(row.get("actual_label", "")).strip()]
    calibrated = [
        row for row in labelled_rows
        if str(row.get("calibration_status", "")).strip().upper() == "CALIBRATING"
    ]
    unknown_rows = [
        row for row in labelled_rows
        if str(row.get("actual_label", "")).strip().upper() == "UNKNOWN"
    ]
    eligible = [
        row for row in labelled_rows
        if _gaze_evaluation_status(row) == "EVALUATED"
    ]
    final_metrics = classification_metrics(
        [str(row["actual_label"]).strip().upper() for row in eligible],
        [str(row["predicted_label"]).strip().upper() for row in eligible],
        GAZE_DIRECTIONAL_CLASSES,
    )
    raw_eligible = [
        row for row in labelled_rows
        if str(row.get("calibration_status", "")).strip().upper() != "CALIBRATING"
        and str(row.get("actual_label", "")).strip().upper()
        in GAZE_DIRECTIONAL_CLASSES
        and _face_present(row.get("face_present"), 0)
        and str(row.get("raw_gaze_label", "")).strip().upper()
        in GAZE_DIRECTIONAL_CLASSES
    ]
    raw_metrics = classification_metrics(
        [str(row["actual_label"]).strip().upper() for row in raw_eligible],
        [str(row["raw_gaze_label"]).strip().upper() for row in raw_eligible],
        GAZE_DIRECTIONAL_CLASSES,
    )
    actual_non_forward = [
        "negative"
        if str(row["actual_label"]).strip().upper() == "FORWARD"
        else "positive"
        for row in eligible
    ]
    predicted_non_forward = [
        "negative"
        if str(row["predicted_label"]).strip().upper() == "FORWARD"
        else "positive"
        for row in eligible
    ]
    predicted_distraction_alert = [
        "positive"
        if bool(row.get("gaze_distraction_confirmed", False))
        else "negative"
        for row in eligible
    ]
    distraction_metrics = {
        "smoothed_non_forward_gaze": classification_metrics(
            actual_non_forward,
            predicted_non_forward,
            ("negative", "positive"),
        ),
        "final_sustained_distraction_alert": classification_metrics(
            actual_non_forward,
            predicted_distraction_alert,
            ("negative", "positive"),
        ),
        "persistence_threshold_seconds": GAZE_DISTRACTION_PERSISTENCE_SECONDS,
        "metric_unit": "labelled frames; not event-level alert matching",
    }
    forward_rows = [row for row in eligible if row["actual_label"] == "FORWARD"]
    forward_up = [
        row for row in forward_rows
        if str(row["predicted_label"]).strip().upper() == "UP"
    ]
    forward_correct = [
        row for row in forward_rows
        if str(row["predicted_label"]).strip().upper() == "FORWARD"
    ]
    unknown_intervals = [
        interval
        for video_name, video_intervals in intervals.items()
        for interval in video_intervals
        if video_name in {str(row["video_name"]) for row in rows}
        and normalize_gaze_label(str(interval["actual_label"])) == "UNKNOWN"
    ]
    unknown_duration = sum(
        float(interval["end_time"]) - float(interval["start_time"])
        for interval in unknown_intervals
    )
    selected_video_names = {str(row["video_name"]) for row in rows}
    unused_ground_truth_videos = sorted(set(intervals) - selected_video_names)
    invalid_predictions = [
        row for row in labelled_rows
        if (
            str(row.get("calibration_status", "")).strip().upper() != "CALIBRATING"
            and str(row.get("actual_label", "")).strip().upper()
            != "UNKNOWN"
            and (
                not _face_present(row.get("face_present"), 0)
                or str(row.get("predicted_label", "")).strip().upper()
                not in GAZE_DIRECTIONAL_CLASSES
            )
        )
    ]

    def mean_coordinate(samples: Sequence[Mapping[str, Any]], field: str) -> float | None:
        values = [
            float(row[field]) for row in samples
            if row.get(field) not in (None, "")
        ]
        return sum(values) / len(values) if values else None

    warnings = (
        ["No valid directional gaze samples were available."]
        if not eligible else []
    )
    if unused_ground_truth_videos:
        warnings.append(
            "Ground-truth videos not included in this run: "
            + ", ".join(unused_ground_truth_videos)
        )

    return {
        **final_metrics,
        "raw_gaze_metrics": raw_metrics,
        "distraction_decision_metrics": distraction_metrics,
        "total_processed_frames": len(rows),
        "labelled_frames": len(labelled_rows),
        "unlabelled_frames_excluded": len(rows) - len(labelled_rows),
        "calibration_frames_processed": sum(
            str(row.get("calibration_status", "")).strip().upper()
            == "CALIBRATING"
            for row in rows
        ),
        "calibration_frames_excluded": len(calibrated),
        "unknown_ground_truth_frames_excluded": len(unknown_rows),
        "unknown_ground_truth_interval_count": len(unknown_intervals),
        "unknown_ground_truth_duration_seconds": unknown_duration,
        "invalid_prediction_frames_excluded": len(invalid_predictions),
        "unused_ground_truth_video_names": unused_ground_truth_videos,
        "evaluated_frames": final_metrics["samples"],
        "forward_predicted_up": {
            "frames": len(forward_up),
            "rate_among_valid_forward_actual": (
                len(forward_up) / len(forward_rows) if forward_rows else 0.0
            ),
            "raw_candidate_frames": sum(
                str(row.get("raw_gaze_label", "")).strip().upper() == "UP"
                for row in forward_rows
            ),
            "raw_candidate_rate": (
                sum(
                    str(row.get("raw_gaze_label", "")).strip().upper() == "UP"
                    for row in forward_rows
                ) / len(forward_rows)
                if forward_rows else 0.0
            ),
            "mean_gaze_y_when_predicted_up": mean_coordinate(forward_up, "gaze_y"),
            "mean_gaze_y_when_predicted_forward": mean_coordinate(
                forward_correct, "gaze_y"
            ),
        },
        "prediction_source": (
            "predicted_label is the live smoothed gaze direction; raw_gaze_label "
            "is the same calibrated classifier before voting/debounce."
        ),
        "warnings": warnings,
    }


def _video_metadata(video_path: Path) -> dict[str, float | int]:
    try:
        import cv2
    except ImportError as error:
        raise _optional_import_error("OpenCV", "monitor") from error
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video metadata: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0 or frame_count <= 0:
            raise EvaluationInputError(
                f"Invalid frame count or FPS metadata for {video_path.name}."
            )
        return {
            "fps": fps,
            "reported_frame_count": frame_count,
            "duration_seconds": frame_count / fps,
        }
    finally:
        capture.release()


def _validate_ground_truth_video_bounds(
    videos: Sequence[Path],
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for video_path in videos:
        duration = float(_video_metadata(video_path)["duration_seconds"])
        for interval in intervals[video_path.name]:
            if float(interval["end_time"]) > duration:
                raise EvaluationInputError(
                    f"Ground-truth interval "
                    f"{interval['line_number']} for {video_path.name!r} ends after "
                    f"the video duration ({duration:.6f}s)."
                )


def _behavior_video_validation(
    videos: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_video[str(row["video_name"])].append(row)

    video_reports = []
    for video_path in videos:
        video_name = video_path.name
        video_rows = rows_by_video[video_name]
        metadata = _video_metadata(video_path)
        timestamps = [float(row["timestamp_seconds"]) for row in video_rows]
        frame_indexes = [int(row["frame_index"]) for row in video_rows]
        monotonic = bool(timestamps) and all(
            later > earlier for earlier, later in zip(timestamps, timestamps[1:])
        )
        contiguous = bool(frame_indexes) and frame_indexes == list(range(len(frame_indexes)))
        timestamps_in_duration = all(
            0.0 <= timestamp < float(metadata["duration_seconds"])
            for timestamp in timestamps
        )
        annotations = []
        within_duration = True
        for interval in intervals[video_name]:
            start = float(interval["start_time"])
            end = float(interval["end_time"])
            label = str(interval["actual_label"]).strip().upper()
            matching_rows = [
                row for row in video_rows
                if start <= float(row["timestamp_seconds"]) < end
            ]
            label_matches = bool(matching_rows) and all(
                str(row["actual_label"]).strip().upper() == label
                for row in matching_rows
            )
            interval_within_duration = end <= float(metadata["duration_seconds"])
            within_duration = within_duration and interval_within_duration
            annotations.append({
                "start_time_seconds": start,
                "end_time_seconds": end,
                "label": label,
                "decoded_frames_in_interval": len(matching_rows),
                "sampled_first_timestamp_seconds": (
                    float(matching_rows[0]["timestamp_seconds"])
                    if matching_rows else None
                ),
                "sampled_last_timestamp_seconds": (
                    float(matching_rows[-1]["timestamp_seconds"])
                    if matching_rows else None
                ),
                "frame_labels_match_interval": label_matches,
                "interval_within_video_duration": interval_within_duration,
            })
        video_reports.append({
            "video_name": video_name,
            "ground_truth_video_name_matches_basename": (
                video_name in intervals
            ),
            "decoded_frames": len(video_rows),
            "metadata_reported_frames": metadata["reported_frame_count"],
            "frame_count_matches_metadata": (
                len(video_rows) == metadata["reported_frame_count"]
            ),
            "fps": metadata["fps"],
            "duration_seconds": metadata["duration_seconds"],
            "first_frame_timestamp_seconds": timestamps[0] if timestamps else None,
            "last_frame_timestamp_seconds": timestamps[-1] if timestamps else None,
            "frame_indexes_contiguous_from_zero": contiguous,
            "frame_timestamps_strictly_increasing": monotonic,
            "frame_timestamps_within_video_duration": timestamps_in_duration,
            "ground_truth_intervals": annotations,
        })
    selected_names = {video.name for video in videos}
    unused_ground_truth_videos = sorted(set(intervals) - selected_names)
    return {
        "videos": video_reports,
        "unused_ground_truth_video_names": unused_ground_truth_videos,
        "all_ground_truth_video_names_match": all(
            report["ground_truth_video_name_matches_basename"]
            for report in video_reports
        ),
        "all_video_timestamps_monotonic": all(
            report["frame_timestamps_strictly_increasing"]
            for report in video_reports
        ),
        "all_frame_indexes_contiguous": all(
            report["frame_indexes_contiguous_from_zero"]
            for report in video_reports
        ),
        "all_timestamps_within_video_durations": all(
            report["frame_timestamps_within_video_duration"]
            for report in video_reports
        ),
        "all_frame_counts_match_metadata": all(
            report["frame_count_matches_metadata"]
            for report in video_reports
        ),
        "all_intervals_within_video_durations": all(
            annotation["interval_within_video_duration"]
            for report in video_reports
            for annotation in report["ground_truth_intervals"]
        ),
        "all_interval_frame_labels_match": all(
            annotation["frame_labels_match_interval"]
            for report in video_reports
            for annotation in report["ground_truth_intervals"]
        ),
    }


def _build_behavior_report(
    videos: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    metrics = _behavior_classification_metrics(rows)
    video_validation = _behavior_video_validation(videos, rows, intervals)
    warnings = [
        "Face-missing frames are measured separately and excluded from the "
        "four-class matrix because the annotations do not independently label "
        "face visibility."
    ]
    unused_ground_truth_videos = video_validation["unused_ground_truth_video_names"]
    if unused_ground_truth_videos:
        warnings.append(
            "Ground-truth videos not included in this run: "
            + ", ".join(unused_ground_truth_videos)
        )
    return {
        "task": "behavior",
        "status": "evaluated",
        "dataset": "video interval annotations",
        "videos": [str(path) for path in videos],
        "prediction_rows": len(rows),
        "sample_counts": {
            "processed_frames": len(rows),
            "labelled_frames": metrics["labelled_frames"],
            "evaluated_frames": metrics["evaluated_frames"],
            "excluded_frames": metrics["excluded_frames"],
        },
        "excluded_samples": {
            "unlabelled": metrics["unlabelled_frames_excluded"],
            "calibration": metrics["calibration_frames_excluded"],
            "face_not_detected": metrics["face_not_detected_excluded_frames"],
            "unclassified": metrics["unclassified_excluded_frames"],
        },
        "metrics": metrics,
        "warnings": warnings,
        "evaluation_protocol": {
            "ground_truth_classes": list(BEHAVIOR_CLASSES),
            "prediction_source": (
                "raw_model_behavior is direct classifier output; predicted_label "
                "is the final rule/temporal-filter decision."
            ),
            "calibration_policy": (
                "Calibration status is copied from the prediction record's "
                "raw_behavior field and those frames are excluded."
            ),
            "unlabelled_policy": (
                "Decoded frames outside all ground-truth intervals are retained "
                "in predictions.csv and excluded from classification metrics."
            ),
            "face_detection_policy": (
                "Face-missing outputs are excluded from class metrics and reported "
                "separately; they are never coerced to NORMAL."
            ),
            "face_expected_assumption": (
                "Frames inside behavior-labeled intervals are treated as frames "
                "where a driver face is expected; the annotations do not provide "
                "independent face-visibility labels."
            ),
        },
        "video_validation": video_validation,
    }


def _resolve_videos(args: argparse.Namespace) -> list[Path]:
    videos = [Path(value).resolve() for value in args.videos]
    if args.video_dir:
        video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
        videos.extend(sorted(
            path.resolve()
            for path in Path(args.video_dir).glob("*")
            if path.is_file() and path.suffix.casefold() in video_extensions
        ))
    if not videos:
        raise EvaluationInputError("Provide at least one video with --videos or --video-dir.")
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Video file does not exist: {missing[0]}")
    unsupported = [path for path in videos if path.suffix.casefold() not in {
        ".avi", ".mp4", ".mov", ".mkv", ".m4v"
    }]
    if unsupported:
        raise EvaluationInputError(f"Not a supported video file: {unsupported[0]}")
    if len({path.name for path in videos}) != len(videos):
        raise EvaluationInputError("Video basenames must be unique for CSV matching.")
    return videos


def _pending_yolo_report(videos: Sequence[Path], reason: str) -> dict[str, Any]:
    return {
        "task": "yolo",
        "status": "pending_annotations",
        "dataset": "video box annotations",
        "videos": [str(path) for path in videos],
        "sample_counts": {
            "videos": len(videos),
            "processed_frames": 0,
            "evaluated_frames": 0,
        },
        "metrics": None,
        "warnings": [reason],
        "interpretation": (
            "No object-detection score is reported until every decoded video "
            "frame has valid ground-truth boxes or an explicit empty-frame row."
        ),
    }


def _write_pending_fatigue_report(
    output_dir: Path,
    dataset: Path | None,
    reason: str,
) -> dict[str, Any]:
    report = {
        "task": "fatigue",
        "status": "pending_data",
        "dataset": str(dataset.resolve()) if dataset is not None else "fatigue data",
        "videos": [],
        "sample_counts": {
            "processed_frames": 0,
            "evaluated_frames": 0,
        },
        "excluded_samples": {},
        "metrics": None,
        "warnings": [reason],
        "interpretation": (
            "No fatigue score is reported until valid labeled clips or an "
            "independent event manifest and test-split metadata are available."
        ),
    }
    write_predictions_csv(
        output_dir / "predictions.csv",
        [],
        fieldnames=["video_name", "frame_index", "timestamp_seconds"],
    )
    write_json(output_dir / "report.json", report)
    write_metrics_csv(output_dir / "report.csv", report)
    stale_confusion_matrix = output_dir / "confusion_matrix.png"
    if stale_confusion_matrix.is_file():
        stale_confusion_matrix.unlink()
    return report


def _write_combined_summary(output_dir: Path) -> dict[str, Any]:
    module_reports: dict[str, Any] = {}
    for task in ("behavior", "gaze", "fatigue", "yolo"):
        report_path = output_dir / task / "report.json"
        if report_path.is_file():
            module_reports[task] = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            module_reports[task] = {
                "task": task,
                "status": "not_run",
                "warnings": [f"No module report found at {report_path}."],
            }
    summary = {
        "status": "summary",
        "modules": module_reports,
        "interpretation": (
            "Module metrics use different evaluation units and are intentionally "
            "not combined into a single accuracy or score."
        ),
    }
    audit_path = output_dir / "threshold-audit.json"
    if audit_path.is_file():
        summary["threshold_audit"] = json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    write_json(output_dir / "combined-summary.json", summary)
    with (output_dir / "combined-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["module", "status", "metric", "value"])
        for task, report in module_reports.items():
            status = report.get("status", "evaluated")
            metrics = report.get("metrics")
            if not isinstance(metrics, Mapping):
                writer.writerow([task, status, "metrics", "not available"])
                continue

            def write_values(prefix: str, values: Mapping[str, Any]) -> None:
                for key, value in values.items():
                    name = f"{prefix}.{key}" if prefix else str(key)
                    if isinstance(value, Mapping):
                        write_values(name, value)
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        writer.writerow([task, status, name, value])

            write_values("", metrics)
    return summary


def _build_threshold_audit(
    output_dir: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    from driver_monitor.config import ProjectPaths
    from driver_monitor.detection import gaze

    paths = ProjectPaths.discover()
    observations = _load_gaze_calibration(
        paths.repository_root, calibration_path.resolve()
    )
    calibrated = gaze.calibrate_gaze_thresholds(observations)
    gaze_threshold_values = {
        name: getattr(calibrated, name)
        for name in (
            "left", "right", "up", "down",
            "left_is_lower", "up_is_lower", "down_is_lower",
        )
    }

    def recorded_performance(task: str, *keys: str) -> Any:
        report_path = output_dir / task / "report.json"
        if not report_path.is_file():
            return {"status": "not_evaluated"}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        values: Any = report.get("metrics", {})
        for key in keys:
            values = values.get(key, {}) if isinstance(values, Mapping) else {}
        return values if values else {"status": report.get("status", "no metrics")}

    rules = [
        {
            "module": "behavior",
            "decision_rule": (
                "Direct classifier output is overridden in priority order by "
                "fatigue, sustained head distraction, yawn, talking, then ML "
                "yawning/talking confidence; final labels pass through a temporal filter."
            ),
            "existing_thresholds": {
                "classifier_confidence": 0.65,
                "yaw_degrees": 25.0,
                "pitch_degrees": 20.0,
                "eye_close_ear": 0.21,
                "eye_open_ear": 0.24,
                "eye_close_confirm_frames": 2,
                "eye_open_confirm_frames": 2,
                "fatigue_eye_closure_seconds": 1.5,
                "fatigue_clear_open_seconds": 0.35,
                "mouth_yawn_mar_threshold": 0.62,
                "mouth_yawn_release_mar": 0.52,
                "yawn_confirmation_frames": 12,
                "mouth_yawn_rearm_seconds": 4.0,
                "talking_window_frames": 24,
                "talking_mar_mean_delta": 0.035,
                "talking_mar_crossings": 2,
                "head_distraction_consecutive_frames": 18,
                "forward_recovery_frames": 8,
                "face_lost_grace_frames": 12,
                "face_landmarker_detection_confidence": 0.6,
                "face_landmarker_tracking_confidence": 0.5,
                "pose_calibration_frames": 45,
                "pose_smoothing_alpha": 0.35,
                "temporal_persistence_seconds": {
                    "talking": 0.35,
                    "yawning": 0.15,
                    "distraction": 0.1,
                    "clear_to_normal": 0.15,
                },
                    "alert_manager_yawn_cooldown_seconds": 10.0,
            },
            "source": [
                "src/driver_monitor/monitoring/legacy_runtime.py",
                "src/driver_monitor/monitoring/temporal.py",
                "Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/train_and_run_driver_detector.py",
            ],
            "configuration_keys": [
                "--confidence", "--pose-yaw", "--pose-pitch",
                "--eye-close-threshold", "--eye-open-threshold",
                "--fatigue-seconds", "--distraction-seconds",
                "BehaviorTemporalFilter.talking_seconds",
                "BehaviorTemporalFilter.yawning_seconds",
                "BehaviorTemporalFilter.distraction_seconds",
                "BehaviorTemporalFilter.clear_seconds",
            ],
            "raw_model_performance": recorded_performance(
                "behavior", "raw_model_metrics"
            ),
            "final_decision_performance": recorded_performance("behavior"),
            "measurements": (
                "Raw classifier and final post-rule/post-temporal classifications "
                "are reported when behavior annotations and runtime dependencies exist."
            ),
            "thresholds_identified": True,
        },
        {
            "module": "gaze",
            "decision_rule": (
                "Calibrated boundaries create raw direction labels; a 5-frame "
                "history requires 4 votes to change the smoothed direction; "
                "integrated distraction requires 0.6 seconds away from FORWARD."
            ),
            "existing_thresholds": {
                "personalized_calibration_boundaries": gaze_threshold_values,
                "horizontal_fallback_offset": 0.05,
                "debounce_history_frames": 5,
                "debounce_required_votes": 4,
                "integrated_distraction_seconds": GAZE_DISTRACTION_PERSISTENCE_SECONDS,
                "calibration_capture_seconds": 1.5,
                "calibration_min_samples": 8,
            },
            "source": [
                "Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/gaze_detection.py",
                "src/driver_monitor/monitoring/legacy_runtime.py",
            ],
            "configuration_keys": [
                "GAZE_HORIZONTAL_THRESHOLD", "GAZE_DEBOUNCE_FRAMES",
                "GAZE_DEBOUNCE_VOTES", "--distraction-seconds",
                "calibrate_gaze_thresholds",
            ],
            "raw_model_performance": recorded_performance(
                "gaze", "raw_gaze_metrics"
            ),
            "final_decision_performance": recorded_performance("gaze"),
            "measurements": (
                "Raw direction versus smoothed direction is reported. The gaze "
                "direction score is distinct from sustained-distraction alert performance."
            ),
            "thresholds_identified": True,
        },
        {
            "module": "fatigue",
            "decision_rule": (
                "Eye closure is debounced before the continuous-closure timer; "
                "the alert clears after eyes reopen continuously for 0.35 seconds."
            ),
            "existing_thresholds": {
                "eye_close_ear": 0.21,
                "eye_open_ear": 0.24,
                "close_confirm_frames": 2,
                "open_confirm_frames": 2,
                "fatigue_confirmation_seconds": 1.5,
                "recovery_open_seconds": 0.35,
                "blink_window_seconds": 10.0,
                "blink_rate_changes_alert": False,
            },
            "source": [
                "Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/train_and_run_driver_detector.py",
                "src/driver_monitor/monitoring/legacy_runtime.py",
            ],
            "configuration_keys": [
                "--eye-close-threshold", "--eye-open-threshold",
                "--fatigue-seconds", "EyeAnalyzer.close_confirm_frames",
                "EyeAnalyzer.open_confirm_frames",
                "EyeAnalyzer.recovery_seconds",
            ],
            "raw_model_performance": {
                "status": "not_exposed: fatigue is a threshold-based eye-state decision"
            },
            "final_decision_performance": recorded_performance("fatigue"),
            "measurements": (
                "Frame-level alert evaluation is reported only for labeled clips; "
                "event precision/recall/delay requires a separate event manifest."
            ),
            "thresholds_identified": True,
        },
        {
            "module": "yolo",
            "decision_rule": (
                "YOLO detections use class-specific confidence filtering; only phone "
                "use alerts add phone tracking, confirmed-driver proximity, and persistence."
            ),
            "existing_thresholds": {
                "image_size": 416,
                "base_confidence": 0.1,
                "nms_iou": 0.45,
                "max_detections": 20,
                "max_box_area_fraction": 0.85,
                "class_confidence": {
                    "Phone": 0.2,
                    "Cigarette": 0.75,
                    "Cup": 0.35,
                    "Bottle": 0.4,
                },
                "worker_frame_skip": 2,
                "history_frames": 3,
                "minimum_hits": 1,
                "phone_track_iou_strictly_greater_than": 0.1,
                "phone_confirmation_seconds": 0.3,
                "phone_minimum_hits": 2,
                "max_missed_frames": 2,
                "stale_track_seconds": 1.0,
                "driver_proximity_multipliers": {
                    "x_left_right_face_width": 0.35,
                    "y_above_face_height": 0.25,
                    "y_below_face_height": 1.25,
                },
                "driver_use_persistence_seconds": 0.75,
                "driver_phone_alert_persistence_seconds": 0.5,
                "driver_identity_confirmations": 2,
                "non_phone_object_alert": "immediate when class-filtered detection is present",
            },
            "source": [
                "src/driver_monitor/monitoring/legacy_runtime.py",
                "src/driver_monitor/detection/objects.py",
                "src/driver_monitor/monitoring/temporal.py",
            ],
            "configuration_keys": [
                "OBJECT_DETECTION_IMAGE_SIZE",
                "OBJECT_DETECTION_BASE_CONFIDENCE",
                "OBJECT_DETECTION_IOU_THRESHOLD",
                "OBJECT_DETECTION_CLASS_CONFIDENCE",
                "OBJECT_DETECTION_FRAME_SKIP",
                "PhoneDetectionValidator.persistence_seconds",
                "PhoneDetectionValidator.use_persistence_seconds",
                "PhoneDetectionValidator.alert_persistence_seconds",
            ],
            "raw_model_performance": recorded_performance(
                "yolo", "raw_model_metrics"
            ),
            "final_decision_performance": recorded_performance("yolo"),
            "measurements": (
                "Raw box and confidence-filtered detector metrics are available "
                "only with complete box labels. Final phone-use alert evaluation "
                "also needs independent driver identity and face-location labels."
            ),
            "thresholds_identified": True,
        },
    ]
    report = {
        "status": "threshold_audit",
        "production_thresholds_modified": False,
        "experimental_threshold_tuning_performed": False,
        "rules": rules,
        "warning": (
            "Threshold values document the current integrated runtime defaults; "
            "they are not optimized or applied by evaluation."
        ),
    }
    write_json(output_dir / "threshold-audit.json", report)
    write_metrics_csv(output_dir / "threshold-audit.csv", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing driver-monitor detectors on recorded videos."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=(
            "behavior", "gaze", "fatigue", "recognition", "emotion", "yolo",
            "summary", "threshold-audit",
        ),
    )
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-output"))
    parser.add_argument("--calibration", type=Path, default=Path("gaze-calibration.json"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--event-manifest", type=Path)
    parser.add_argument("--split-metadata", type=Path)
    parser.add_argument("--pending-reason")
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--resize-scale", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "summary":
        report = _write_combined_summary(args.output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.task == "threshold-audit":
        report = _build_threshold_audit(args.output_dir, args.calibration)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.pending_reason is not None:
        if args.task != "fatigue":
            raise EvaluationInputError(
                "--pending-reason is supported only with --task fatigue."
            )
        if not args.pending_reason.strip():
            raise EvaluationInputError("--pending-reason must not be empty.")
        if args.ground_truth is not None and not args.ground_truth.is_file():
            raise FileNotFoundError(
                f"Ground-truth file does not exist: {args.ground_truth}"
            )
        report = _write_pending_fatigue_report(
            args.output_dir,
            args.ground_truth,
            args.pending_reason.strip(),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.task == "fatigue" and args.event_manifest is not None:
        if args.model is None or args.split_metadata is None:
            raise EvaluationInputError(
                "Event-based fatigue evaluation requires --model and --split-metadata."
            )
        report, rows = evaluate_fatigue_events(
            args.event_manifest.parent,
            {
                "events_csv": args.event_manifest.resolve(),
                "model_path": args.model.resolve(),
                "split_metadata": args.split_metadata.resolve(),
            },
        )
        report["task"] = "fatigue"
        report["prediction_rows"] = len(rows)
        write_predictions_csv(args.output_dir / "predictions.csv", rows)
        write_json(args.output_dir / "report.json", report)
        write_metrics_csv(args.output_dir / "report.csv", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    videos = _resolve_videos(args)
    if args.task == "yolo":
        if args.annotations is None:
            rows = []
            report = _pending_yolo_report(
                videos,
                "No bounding-box annotation file was supplied; evaluation is "
                "pending complete frame/object annotations.",
            )
            metrics = None
            confusion = None
            labels = None
        else:
            if args.model is None:
                from driver_monitor.config import ProjectPaths

                args.model = ProjectPaths.discover().object_model
            rows, metrics = _yolo_rows(
                videos, args.model.resolve(), args.annotations.resolve()
            )
            report = {
                "task": args.task,
                "status": "evaluated",
                "dataset": str(args.annotations.resolve()),
                "videos": [str(path) for path in videos],
                "sample_counts": {
                    "videos": len(videos),
                    "processed_frames": metrics["images"],
                    "evaluated_frames": metrics["images"],
                },
                "excluded_samples": {},
                "metrics": metrics,
                "warnings": [
                    metrics["final_alert_metrics_status"],
                    *(
                        ["No annotated object instances are present; AP is not informative."]
                        if not any(
                            item["ground_truth"]
                            for item in metrics["per_class"].values()
                        )
                        else []
                    ),
                ],
                "metric_protocol": {
                    "precision_recall_iou_threshold": 0.5,
                    "map_thresholds": [
                        round(0.5 + step * 0.05, 2) for step in range(10)
                    ],
                    "description": (
                        "Raw boxes use a low inference floor; post-class-confidence "
                        "boxes use the live class filters and 0.85 max-box fraction. "
                        "scored separately; temporal/driver-proximity alerts are "
                        "not inferred from object labels."
                    ),
                },
            }
            confusion = metrics["confusion_matrix"]
            labels = metrics["confusion_labels"]
    else:
        if args.ground_truth is None:
            raise EvaluationInputError("--ground-truth is required for this task.")
        intervals = load_ground_truth(args.ground_truth.resolve())
        for video in videos:
            if video.name not in intervals:
                raise EvaluationInputError(
                    f"No ground-truth intervals found for video {video.name!r}."
                )
        _validate_ground_truth_video_bounds(videos, intervals)
        if args.task in {"behavior", "gaze", "fatigue"}:
            rows = [
                row
                for video in videos
                for row in _behavior_rows(
                    video,
                    args.calibration.resolve(),
                    args.task,
                    intervals,
                )
            ]
        elif args.task == "recognition":
            if args.database is None:
                from driver_monitor.config import ProjectPaths

                args.database = ProjectPaths.discover().driver_database
            rows = [
                row
                for video in videos
                for row in _recognition_rows(
                    video,
                    args.database.resolve(),
                    intervals,
                    args.tolerance,
                    args.resize_scale,
                )
            ]
        else:
            rows = [
                row
                for video in videos
                for row in _emotion_rows(video, intervals)
            ]
        if args.task == "behavior":
            report = _build_behavior_report(videos, rows, intervals)
            metrics = report["metrics"]
        elif args.task == "gaze":
            metrics = _gaze_classification_metrics(rows, intervals)
            report = {
                "task": "gaze",
                "status": "evaluated",
                "dataset": str(args.ground_truth.resolve()),
                "videos": [str(path) for path in videos],
                "sample_counts": {
                    "processed_frames": len(rows),
                    "labelled_frames": metrics["labelled_frames"],
                    "evaluated_frames": metrics["evaluated_frames"],
                },
                "excluded_samples": {
                    "unlabelled": metrics["unlabelled_frames_excluded"],
                    "calibration": metrics["calibration_frames_excluded"],
                    "unknown_ground_truth": metrics[
                        "unknown_ground_truth_frames_excluded"
                    ],
                    "invalid_predictions": metrics[
                        "invalid_prediction_frames_excluded"
                    ],
                },
                "metrics": metrics,
                "warnings": metrics["warnings"],
                "interpretation": (
                    "Directional metrics use FORWARD/RIGHT/LEFT/UP/DOWN only. "
                    "UNKNOWN ground-truth intervals and invalid predictions are "
                    "excluded and counted separately."
                ),
            }
        else:
            metrics = _evaluate_classification(rows, args.task)
            report = {
                "task": args.task,
                "status": "evaluated",
                "dataset": str(args.ground_truth.resolve()),
                "videos": [str(path) for path in videos],
                "sample_counts": {
                    "processed_frames": len(rows),
                    "labelled_frames": metrics.get("labelled_frames", len(rows)),
                    "evaluated_frames": metrics["samples"],
                },
                "excluded_samples": {
                    "unlabelled": metrics.get("unlabelled_frames_excluded", 0),
                    "calibration": metrics.get("calibration_frames_excluded", 0),
                    "face_not_detected": metrics.get(
                        "face_not_detected_frames_excluded", 0
                    ),
                },
                "metrics": metrics,
                "warnings": (
                    [
                        "Frame-level fatigue results do not measure event matching, "
                        "detection delay, or missed fatigue events."
                    ]
                    if args.task == "fatigue" and args.event_manifest is None
                    else []
                ),
            }
        if args.task == "recognition":
            identities = [
                row for row in rows
                if row.get("actual_label") not in (None, "")
            ]
            unknown_count = sum(
                str(row["actual_label"]).casefold() == "unknown"
                for row in identities
            )
            known_count = len(identities) - unknown_count
            false_accepts = sum(
                str(row["actual_label"]).casefold() == "unknown"
                and str(row["predicted_label"]).casefold() != "unknown"
                for row in identities
            )
            false_rejects = sum(
                str(row["actual_label"]).casefold() != "unknown"
                and str(row["predicted_label"]).casefold() == "unknown"
                for row in identities
            )
            report["identity_metrics"] = {
                "known_samples": known_count,
                "unknown_samples": unknown_count,
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "far": false_accepts / unknown_count if unknown_count else 0.0,
                "frr": false_rejects / known_count if known_count else 0.0,
                "known_identity_accuracy": sum(
                    str(row["actual_label"]).casefold()
                    == str(row["predicted_label"]).casefold()
                    for row in identities
                    if str(row["actual_label"]).casefold() != "unknown"
                ) / known_count if known_count else 0.0,
                "unknown_rejection_rate": (
                    (unknown_count - false_accepts) / unknown_count
                    if unknown_count else 0.0
                ),
            }
        confusion = metrics.get("confusion_matrix")
        labels = (
            list(BEHAVIOR_CLASSES)
            if args.task == "behavior"
            else list(GAZE_DIRECTIONAL_CLASSES)
            if args.task == "gaze"
            else list(confusion) if isinstance(confusion, Mapping) else None
        )

    report["prediction_rows"] = len(rows)
    prediction_fields = None
    if args.task == "yolo" and not rows:
        prediction_fields = [
            "video_name", "frame_index", "timestamp_seconds",
            "prediction_stage", "predicted_class", "confidence",
            "x_min", "y_min", "x_max", "y_max",
        ]
    write_predictions_csv(
        args.output_dir / "predictions.csv", rows, fieldnames=prediction_fields
    )
    write_json(args.output_dir / "report.json", report)
    write_metrics_csv(args.output_dir / "report.csv", report)
    if confusion and labels:
        write_confusion_matrix_png(
            args.output_dir / "confusion_matrix.png",
            confusion,
            labels,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
