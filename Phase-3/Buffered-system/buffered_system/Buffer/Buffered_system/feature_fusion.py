"""Feature fusion for the existing face recognition and monitoring pipeline.

This module intentionally consumes the values already produced by the existing
face-recognition and behavior workers from the shared ResultBuffer and converts
those numerical signals into one fixed-length feature vector per frame.

Design notes:
- No YOLO/object-detection features are included yet.
- Driver IDs remain metadata; the vector itself contains only numeric values.
- Missing values are replaced with safe defaults so the pipeline does not crash.
- The feature order is fixed by FEATURE_NAMES so downstream stages (LSTM later)
  can depend on a stable vector layout.
"""

import math
import time
from typing import Any, Dict, Iterable, List

FEATURE_NAMES = [
    'recognition_confidence',
    'recognition_status',
    'left_ear',
    'right_ear',
    'average_ear',
    'blink_count_10s',
    'blink_rate',
    'eyes_closed_duration',
    'mar',
    'pitch',
    'yaw',
    'roll',
    'fatigue_alert',
    'distracted',
    'yawning',
    'yawn_event',
    'talking',
    'behavior_confidence',
]

DEFAULT_NUMERIC = 0.0


def _safe_float(value: Any, default: float = DEFAULT_NUMERIC) -> float:
    """Normalize missing or non-numeric values to a safe float default."""
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return float(default)
    if math.isnan(numeric) or math.isinf(numeric):
        return float(default)
    return float(numeric)


def _as_binary_flag(value: Any, default: float = DEFAULT_NUMERIC) -> float:
    """Convert bool-like or numeric status fields to a 0.0/1.0 float."""
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip().lower()
        if text in {'true', 'yes', 'y', '1'}:
            return 1.0
        if text in {'false', 'no', 'n', '0', ''}:
            return 0.0
        try:
            numeric = float(text)
        except ValueError:
            return float(default)
    if math.isnan(numeric) or math.isinf(numeric):
        return float(default)
    return 1.0 if numeric > 0.0 else 0.0


def _recognition_status(values: Dict[str, Any]) -> float:
    if values.get('driver_confirmed') is not None:
        return _as_binary_flag(values.get('driver_confirmed'), 0.0)
    if values.get('recognition_complete') is not None:
        return _as_binary_flag(values.get('recognition_complete'), 0.0)
    if values.get('driver_name') not in {None, 'UNKNOWN', 'IDENTIFYING...'}:
        return 1.0
    return 0.0


def _build_feature_vector(values: Dict[str, Any]) -> List[float]:
    vector = [
        _safe_float(values.get('recognition_confidence'), 0.0),
        _recognition_status(values),
        _safe_float(values.get('left_ear'), 0.0),
        _safe_float(values.get('right_ear'), 0.0),
        _safe_float(values.get('average_ear'), 0.0),
        _safe_float(values.get('blink_count_10s'), 0.0),
        _safe_float(values.get('blink_rate'), 0.0),
        _safe_float(values.get('eyes_closed_duration'), 0.0),
        _safe_float(values.get('mar'), 0.0),
        _safe_float(values.get('pitch'), 0.0),
        _safe_float(values.get('yaw'), 0.0),
        _safe_float(values.get('roll'), 0.0),
        _as_binary_flag(values.get('fatigue_alert'), 0.0),
        _as_binary_flag(values.get('distracted'), 0.0),
        _as_binary_flag(values.get('yawning'), 0.0),
        _as_binary_flag(values.get('yawn_event'), 0.0),
        _as_binary_flag(values.get('talking'), 0.0),
        _safe_float(values.get('behavior_confidence'), 0.0),
    ]
    return vector


def validate_feature_vector(feature_vector: Iterable[float], expected_names: Iterable[str]) -> None:
    vector = list(feature_vector)
    expected_count = len(list(expected_names))
    if len(vector) != expected_count:
        raise ValueError(
            f'Feature vector length mismatch: expected {expected_count}, got {len(vector)}'
        )
    for index, value in enumerate(vector):
        if isinstance(value, bool):
            raise TypeError(
                f'Boolean values are not valid numeric features at index {index}: {value}')
        if not isinstance(value, (int, float)):
            raise TypeError(
                f'Feature value at index {index} is not numeric: {value!r} ({type(value).__name__})'
            )
        if math.isnan(float(value)) or math.isinf(float(value)):
            raise ValueError(
                f'Feature value at index {index} is NaN/Inf: {value!r}')


def fuse_frame_features(values: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the current face-recognition and face-monitoring metrics.

    The returned structure keeps driver_id/timestamp as metadata while returning a
    fixed-length numeric vector suitable for later model stages.
    """
    if values is None:
        values = {}

    driver_id = str(values.get('driver_name') or 'UNKNOWN')
    feature_vector = _build_feature_vector(values)
    validate_feature_vector(feature_vector, FEATURE_NAMES)

    print(
        f'[FEATURE FUSION] Driver: {driver_id} | Feature count: {len(feature_vector)} | '
        f'Feature vector: {feature_vector}'
    )

    timestamp = values.get('behavior_timestamp')
    if timestamp is None:
        timestamp = values.get('recognition_timestamp')
    if timestamp is None:
        timestamp = time.time()

    return {
        'driver_id': driver_id,
        'feature_vector': feature_vector,
        'feature_names': list(FEATURE_NAMES),
        'timestamp': float(timestamp),
    }


class FeatureFusion:
    """Small wrapper to make later YOLO integration easy without changing the pipeline."""

    def __init__(self):
        self.feature_names = list(FEATURE_NAMES)

    def fuse(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return fuse_frame_features(values)
