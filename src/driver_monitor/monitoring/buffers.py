from __future__ import annotations

import threading
from typing import Generic, TypeVar

from driver_monitor.models import FramePacket


T = TypeVar("T")


class LatestFrameBuffer(Generic[T]):
    """A single-slot frame buffer that replaces work that has become stale."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: T | None = None

    def put(self, packet: T) -> None:
        with self._condition:
            self._packet = packet
            self._condition.notify_all()

    def get_newest(
        self, last_frame_id: int, stop_event: threading.Event
    ) -> T | None:
        with self._condition:
            while (
                self._packet is None
                or getattr(self._packet, "frame_id", -1) <= last_frame_id
            ) and not stop_event.is_set():
                self._condition.wait(timeout=0.1)
            if stop_event.is_set():
                return None
            return self._packet

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._packet = None


class ResultBuffer:
    """Thread-safe latest-result storage with independent detector namespaces."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._results: dict[str, object] = {}

    def update(self, source: str, result: object) -> None:
        if not source:
            raise ValueError("Result source must be a non-empty string")
        with self._lock:
            self._results[source] = result

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._results.copy()

    def clear(self) -> None:
        with self._lock:
            self._results.clear()

    def clear_source(self, source: str) -> None:
        with self._lock:
            self._results.pop(source, None)


def frame_packet(
    frame_id: int, captured_at: float, frame: T
) -> FramePacket:
    return FramePacket(frame_id, captured_at, frame)
