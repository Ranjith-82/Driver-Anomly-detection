from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping


FEATURE_NAMES = [
    "recognition_confidence",
    "recognition_status",
    "left_ear",
    "right_ear",
    "average_ear",
    "blink_count_10s",
    "blink_rate",
    "eyes_closed_duration",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "fatigue_alert",
    "distracted",
    "yawning",
    "yawn_event",
    "talking",
    "behavior_confidence",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _as_binary_flag(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return 1.0
    if text in {"false", "no", "n", "0", ""}:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return 1.0 if numeric > 0.0 else 0.0


def _recognition_status(values: Mapping[str, object]) -> float:
    if values.get("driver_confirmed") is not None:
        return _as_binary_flag(values.get("driver_confirmed"))
    if values.get("recognition_complete") is not None:
        return _as_binary_flag(values.get("recognition_complete"))
    if values.get("driver_name") not in {None, "UNKNOWN", "IDENTIFYING..."}:
        return 1.0
    return 0.0


def build_feature_vector(
    values: Mapping[str, object] | None,
) -> list[float]:
    values = values or {}
    vector = [
        _safe_float(values.get("recognition_confidence")),
        _recognition_status(values),
        _safe_float(values.get("left_ear")),
        _safe_float(values.get("right_ear")),
        _safe_float(values.get("average_ear")),
        _safe_float(values.get("blink_count_10s")),
        _safe_float(values.get("blink_rate")),
        _safe_float(values.get("eyes_closed_duration")),
        _safe_float(values.get("mar")),
        _safe_float(values.get("pitch")),
        _safe_float(values.get("yaw")),
        _safe_float(values.get("roll")),
        _as_binary_flag(values.get("fatigue_alert")),
        _as_binary_flag(values.get("distracted")),
        _as_binary_flag(values.get("yawning")),
        _as_binary_flag(values.get("yawn_event")),
        _as_binary_flag(values.get("talking")),
        _safe_float(values.get("behavior_confidence")),
    ]
    validate_feature_vector(vector, FEATURE_NAMES)
    return vector


def validate_feature_vector(
    feature_vector: Iterable[float], expected_names: Iterable[str]
) -> None:
    vector = list(feature_vector)
    names = list(expected_names)
    if len(vector) != len(names):
        raise ValueError(
            f"Feature vector length mismatch: expected {len(names)}, got {len(vector)}"
        )
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Feature value at index {index} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"Feature value at index {index} must be finite")


def fuse_frame_features(
    values: Mapping[str, object] | None,
) -> dict[str, object]:
    values = values or {}
    timestamp = values.get("behavior_timestamp")
    if timestamp is None:
        timestamp = values.get("recognition_timestamp")
    if timestamp is None:
        timestamp = time.time()
    return {
        "driver_id": str(values.get("driver_name") or "UNKNOWN"),
        "feature_vector": build_feature_vector(values),
        "feature_names": list(FEATURE_NAMES),
        "timestamp": float(timestamp),
    }
