from __future__ import annotations

from collections.abc import Mapping
import math

import cv2


EMOTION_EMOJIS = {
    "angry": "😠",
    "confused": "😕",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "shy": "😳",
    "surprise": "😲",
    "unknown": "❓",
}


def draw_detection_annotation(
    frame,
    box,
    label: str,
    color: tuple[int, int, int],
    occupied_labels: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[int, int, int, int] | None:
    """Draw a clipped detection box and place its label without collisions."""
    height, width = frame.shape[:2]
    if len(box) != 4:
        raise ValueError("Detection boxes must contain four coordinates")
    x1, y1, x2, y2 = (int(value) for value in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    padding = 4
    label_width = text_width + padding * 2
    label_height = text_height + baseline + padding * 2
    if label_width > width:
        scale *= max(0.25, (width - padding * 2) / max(text_width, 1))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        label_width = min(width, text_width + padding * 2)
        label_height = text_height + baseline + padding * 2

    candidates = (
        (x1, y1 - label_height - 3),
        (x1, y2 + 3),
        (x2 - label_width, y1 - label_height - 3),
        (x2 - label_width, y2 + 3),
    )
    blocked_regions = tuple(occupied_labels) + (
        (x1, y1, x2 + 1, y2 + 1),
    )
    placed = []
    for candidate_x, candidate_y in candidates:
        left = max(0, min(width - label_width, candidate_x))
        top = max(0, min(height - label_height, candidate_y))
        bounds = (left, top, left + label_width, top + label_height)
        overlap = sum(
            max(0, min(bounds[2], other[2]) - max(bounds[0], other[0]))
            * max(0, min(bounds[3], other[3]) - max(bounds[1], other[1]))
            for other in blocked_regions
        )
        placed.append((overlap, len(placed), bounds))
    _, _, bounds = min(placed)
    left, top, right, bottom = bounds
    cv2.rectangle(frame, (left, top), (right, bottom), (12, 18, 26), -1)
    cv2.putText(
        frame,
        label,
        (left + padding, top + padding + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return bounds


def emotion_display(values: Mapping[str, object]) -> tuple[str, str]:
    status = str(values.get("emotion_status", "uncertain")).casefold()
    label = str(values.get("emotion_label", "Uncertain"))
    if status == "ready":
        return label, "ready"
    if status == "unknown":
        return "unknown", "unknown"
    if status == "no_face":
        return "No face", "no_face"
    if status == "unavailable":
        return "Uncertain", "unavailable"
    if status == "stale":
        return "Uncertain", "stale"
    return "Uncertain", "uncertain"


def emotion_emoji(label: str) -> str:
    return EMOTION_EMOJIS.get(label.casefold(), EMOTION_EMOJIS["unknown"])


def emotion_confidence_text(values: Mapping[str, object]) -> str:
    if str(values.get("emotion_status", "")).casefold() != "ready":
        return ""
    score = values.get("emotion_confidence")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        return ""
    return f"Score: {float(score) * 100:.1f}%"
