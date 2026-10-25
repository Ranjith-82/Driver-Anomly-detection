from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from driver_monitor.models import (
    DetectorStatus,
    EmotionEstimate,
    EmotionLabel,
)


SOURCE_TO_LABEL = {
    "angry": EmotionLabel.ANGER,
    "happy": EmotionLabel.SMILING,
    "neutral": EmotionLabel.NEUTRAL,
}
DISPLAY_TO_SOURCE = {
    EmotionLabel.ANGER.value: "angry",
    EmotionLabel.SMILING.value: "happy",
    EmotionLabel.NEUTRAL.value: "neutral",
}
MINIMUM_CLASS_PRECISION = 0.75


@dataclass(frozen=True)
class EmotionThresholdPolicy:
    thresholds: Mapping[str, float]
    evaluation_version: str

    @classmethod
    def load(cls, path: Path) -> "EmotionThresholdPolicy":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Emotion threshold file must contain a JSON object")
        version = data.get("evaluation_version")
        classes = data.get("classes")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Emotion threshold file is missing evaluation_version")
        if not isinstance(classes, dict):
            raise ValueError("Emotion threshold file is missing classes")
        thresholds: dict[str, float] = {}
        for source_label in SOURCE_TO_LABEL:
            entry = classes.get(source_label)
            if not isinstance(entry, dict) or entry.get("enabled") is not True:
                continue
            threshold = entry.get("threshold")
            validation_precision = entry.get("validation_precision")
            held_out_precision = entry.get("held_out_precision")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or isinstance(validation_precision, bool)
                or not isinstance(validation_precision, (int, float))
                or not math.isfinite(float(validation_precision))
                or isinstance(held_out_precision, bool)
                or not isinstance(held_out_precision, (int, float))
                or not math.isfinite(float(held_out_precision))
                or validation_precision < MINIMUM_CLASS_PRECISION
                or held_out_precision < MINIMUM_CLASS_PRECISION
            ):
                continue
            thresholds[source_label] = float(threshold)
        return cls(thresholds, version)


def classify_scores(
    scores: Mapping[str, object],
    thresholds: Mapping[str, float | None],
    *,
    frame_id: int | None = None,
    captured_at: float | None = None,
    completed_at: float | None = None,
) -> EmotionEstimate:
    if not scores:
        return EmotionEstimate(
            EmotionLabel.UNCERTAIN,
            DetectorStatus.NO_FACE,
            frame_id=frame_id,
            captured_at=captured_at,
            completed_at=completed_at,
        )

    clean_scores: dict[str, float] = {}
    for source_label, raw_score in scores.items():
        if (
            not isinstance(source_label, str)
            or isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            return EmotionEstimate(
                EmotionLabel.UNCERTAIN,
                DetectorStatus.UNCERTAIN,
                frame_id=frame_id,
                captured_at=captured_at,
                completed_at=completed_at,
            )
        clean_scores[source_label.casefold()] = float(raw_score)
    if not clean_scores:
        return EmotionEstimate(EmotionLabel.UNCERTAIN, DetectorStatus.UNCERTAIN)

    source_label = max(clean_scores, key=clean_scores.get)
    label = SOURCE_TO_LABEL.get(source_label)
    threshold = thresholds.get(source_label) if label is not None else None
    score = clean_scores[source_label]
    if label is None or threshold is None or score < threshold:
        return EmotionEstimate(
            EmotionLabel.UNCERTAIN,
            DetectorStatus.UNCERTAIN,
            source_label=source_label,
            frame_id=frame_id,
            captured_at=captured_at,
            completed_at=completed_at,
        )
    return EmotionEstimate(
        label,
        DetectorStatus.READY,
        source_label=source_label,
        score=score,
        frame_id=frame_id,
        captured_at=captured_at,
        completed_at=completed_at,
    )


class DeepFaceEmotionAdapter:
    """Lazy DeepFace adapter that requests emotion inference only."""

    def __init__(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as error:
            raise RuntimeError(
                "DeepFace is not installed. Install the project emotion extra."
            ) from error
        self._deepface = DeepFace

    def warm_up(self) -> None:
        self._deepface.build_model("Emotion")

    def analyze_face_crop(self, face_crop: object) -> Mapping[str, object]:
        result = self._deepface.analyze(
            img_path=face_crop,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            if not result:
                return {}
            result = result[0]
        if not isinstance(result, dict):
            raise ValueError("DeepFace returned an unsupported emotion result")
        scores = result.get("emotion")
        if not isinstance(scores, dict):
            raise ValueError("DeepFace result did not contain emotion scores")
        return scores
