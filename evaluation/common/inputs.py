from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


class EvaluationInputError(ValueError):
    """Raised when an evaluation manifest is missing or contains invalid data."""


def parse_video_time(value: str) -> float:
    value = value.strip()
    if not value:
        raise EvaluationInputError("Video time must not be empty")
    try:
        if ":" not in value:
            seconds = float(value)
        else:
            parts = value.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            numbers = [float(part) for part in parts]
            if any(not math.isfinite(number) or number < 0 for number in numbers):
                raise ValueError
            if len(numbers) == 2:
                minutes, seconds_part = numbers
                if not minutes.is_integer() or seconds_part >= 60:
                    raise ValueError
                seconds = minutes * 60.0 + seconds_part
            else:
                hours, minutes, seconds_part = numbers
                if (
                    not hours.is_integer()
                    or not minutes.is_integer()
                    or minutes >= 60
                    or seconds_part >= 60
                ):
                    raise ValueError
                seconds = hours * 3600.0 + minutes * 60.0 + seconds_part
    except ValueError as error:
        raise EvaluationInputError(f"Invalid video time: {value!r}") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise EvaluationInputError(f"Invalid video time: {value!r}")
    return seconds


GAZE_LABELS = ("FORWARD", "RIGHT", "LEFT", "UP", "DOWN", "UNKNOWN")
GAZE_GROUND_TRUTH_LABELS = {
    "STRAIGHT": "FORWARD",
    "FORWARD": "FORWARD",
    "RIGHT": "RIGHT",
    "LEFT": "LEFT",
    "LOOKING_UP": "UP",
    "UP": "UP",
    "UNKNOWN": "UNKNOWN",
}


def normalize_gaze_label(label: str) -> str:
    normalized = label.strip().upper()
    try:
        return GAZE_GROUND_TRUTH_LABELS[normalized]
    except KeyError as error:
        raise EvaluationInputError(
            f"Unsupported gaze ground-truth label {label!r}; expected one of "
            f"{tuple(GAZE_GROUND_TRUTH_LABELS)}."
        ) from error


def load_ground_truth(path: Path) -> dict[str, list[dict[str, Any]]]:
    required = {"video_name", "start_time", "end_time", "actual_label"}
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise EvaluationInputError(
                f"Ground-truth CSV is missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("video_name") or "").strip()
            label = (row.get("actual_label") or "").strip()
            if not name or not label:
                raise EvaluationInputError(
                    f"Ground-truth row {line_number} requires video_name and actual_label"
                )
            start = parse_video_time(row["start_time"])
            end = parse_video_time(row["end_time"])
            if end <= start:
                raise EvaluationInputError(
                    f"Ground-truth row {line_number} has end_time <= start_time"
                )
            by_video[name].append({
                "start_time": start,
                "end_time": end,
                "actual_label": label,
                "line_number": line_number,
            })

    for name, intervals in by_video.items():
        intervals.sort(key=lambda interval: interval["start_time"])
        for previous, current in zip(intervals, intervals[1:]):
            if current["start_time"] < previous["end_time"]:
                raise EvaluationInputError(
                    f"Ground-truth intervals overlap for {name!r} "
                    f"on rows {previous['line_number']} and {current['line_number']}"
                )
    return dict(by_video)


def label_at(
    intervals: Mapping[str, Sequence[Mapping[str, Any]]],
    video_name: str,
    timestamp: float,
) -> str | None:
    for interval in intervals.get(video_name, ()):
        if interval["start_time"] <= timestamp < interval["end_time"]:
            return str(interval["actual_label"])
    return None


def normalize_binary_label(label: str) -> str:
    normalized = label.strip().casefold()
    if normalized in {"1", "true", "yes", "positive", "drowsy", "alert"}:
        return "positive"
    if normalized in {
        "0", "false", "no", "negative", "normal", "not_drowsy", "clear"
    }:
        return "negative"
    raise EvaluationInputError(
        f"Expected a binary label (positive/negative or true/false), got {label!r}"
    )


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> None:
    declared = set(classes)
    for index, row in enumerate(rows, start=1):
        for field in ("actual", "predicted"):
            label = row.get(field)
            if label not in declared:
                raise EvaluationInputError(
                    f"Prediction row {index} has undeclared {field} label {label!r}"
                )


def verify_independent_split(
    metadata_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationInputError(
            f"Could not read split metadata: {metadata_path}"
        ) from error

    if metadata.get("independent_test") is not True:
        raise EvaluationInputError("Split metadata does not declare an independent test set")
    training_ids = set(metadata.get("training_identity_ids", ()))
    test_ids = set(metadata.get("test_identity_ids", ()))
    overlap = training_ids & test_ids
    if overlap:
        raise EvaluationInputError(
            f"Identity leakage between training and test sets: {sorted(overlap)}"
        )
    training_hashes = set(metadata.get("training_sha256", ()))
    test_hashes = set(metadata.get("test_sha256", ()))
    hash_overlap = training_hashes & test_hashes
    if hash_overlap:
        raise EvaluationInputError(
            f"File hash leakage between training and test sets: {len(hash_overlap)} overlap"
        )
    row_ids = {str(row.get("identity_id", "")) for row in rows}
    row_ids.discard("")
    unexpected_ids = row_ids - test_ids
    if unexpected_ids:
        raise EvaluationInputError(
            f"Evaluation rows contain identities outside the test split: "
            f"{sorted(unexpected_ids)}"
        )
    row_hashes = {str(row.get("sha256", "")) for row in rows}
    row_hashes.discard("")
    unexpected_hashes = row_hashes - test_hashes
    if unexpected_hashes:
        raise EvaluationInputError(
            f"Evaluation rows contain file hashes outside the test split: "
            f"{sorted(unexpected_hashes)}"
        )

