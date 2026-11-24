from __future__ import annotations

from collections.abc import Iterator
import math
from pathlib import Path
from typing import Any


def iter_video_frames(video_path: Path) -> Iterator[tuple[int, float, Any]]:
    """Yield decoded frames with monotonically increasing video-time seconds."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to read videos; install the project monitor extra."
        ) from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video has invalid FPS metadata: {video_path}")

    frame_index = 0
    previous_timestamp = -1.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            reported_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = reported_ms / 1000.0
            if (
                not timestamp >= 0
                or timestamp <= previous_timestamp
                or (frame_index > 0 and timestamp == 0)
            ):
                timestamp = frame_index / fps
            if timestamp <= previous_timestamp:
                timestamp = previous_timestamp + 1.0 / fps
            yield frame_index, timestamp, frame
            previous_timestamp = timestamp
            frame_index += 1
        if frame_index == 0:
            raise OSError(f"Video contains no decodable frames: {video_path}")
    finally:
        capture.release()
