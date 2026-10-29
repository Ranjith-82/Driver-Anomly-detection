from __future__ import annotations

import threading
import time
from typing import Any

from driver_monitor.models import FramePacket
from driver_monitor.monitoring.buffers import LatestFrameBuffer
from driver_monitor.monitoring.timing import log_timing


class CameraProducer(threading.Thread):
    def __init__(
        self,
        frame_buffer: LatestFrameBuffer[FramePacket],
        stop_event: threading.Event,
        camera_index: int = 0,
        width: int = 960,
        height: int = 540,
    ) -> None:
        super().__init__(name="camera-producer", daemon=True)
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.error: str | None = None
        self._capture: Any = None

    def run(self) -> None:
        try:
            import cv2

            log_timing(
                "CAMERA_OPEN_START",
                camera_index=self.camera_index,
            )
            self._capture = cv2.VideoCapture(self.camera_index)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.camera_index}")
            log_timing("CAMERA_READY", camera_index=self.camera_index)
            frame_id = 0
            while not self.stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError("Camera failed to return a frame")
                if frame_id == 0:
                    log_timing(
                        "CAMERA_FIRST_FRAME",
                        frame_id=frame_id,
                        shape=frame.shape,
                    )
                self.frame_buffer.put(
                    FramePacket(frame_id, time.monotonic(), frame)
                )
                frame_id += 1
        except Exception as error:
            self.error = str(error)
            log_timing("CAMERA_FAILED", error=self.error)
            self.stop_event.set()
        finally:
            if self._capture is not None:
                self._capture.release()
