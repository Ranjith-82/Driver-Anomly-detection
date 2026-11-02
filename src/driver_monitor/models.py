from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DetectorStatus(StrEnum):
    READY = "ready"
    UNCERTAIN = "uncertain"
    NO_FACE = "no_face"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class EmotionLabel(StrEnum):
    ANGER = "anger"
    SMILING = "smiling (estimated from happy)"
    NEUTRAL = "neutral"
    UNCERTAIN = "Uncertain"


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_at: float
    frame: Any

    @property
    def timestamp(self) -> float:
        return self.captured_at


@dataclass(frozen=True)
class DetectionResult:
    source: str
    status: DetectorStatus
    frame_id: int | None = None
    captured_at: float | None = None
    completed_at: float | None = None
    payload: Any = None
    error_summary: str | None = None


@dataclass(frozen=True)
class EmotionEstimate:
    label: EmotionLabel
    status: DetectorStatus
    source_label: str | None = None
    score: float | None = None
    frame_id: int | None = None
    captured_at: float | None = None
    completed_at: float | None = None
    error_summary: str | None = None
