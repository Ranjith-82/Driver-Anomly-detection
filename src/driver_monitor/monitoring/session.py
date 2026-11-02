from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from driver_monitor.monitoring.buffers import LatestFrameBuffer, ResultBuffer


class SessionState(StrEnum):
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ENDED = "ended"


@dataclass
class LiveMonitoringSession:
    """Owns transient detector state for exactly one local camera session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.monotonic)
    state: SessionState = SessionState.STARTING
    stop_event: threading.Event = field(default_factory=threading.Event)
    frames: LatestFrameBuffer = field(default_factory=LatestFrameBuffer)
    results: ResultBuffer = field(default_factory=ResultBuffer)
    workers: list[threading.Thread] = field(default_factory=list)

    def start(self, workers: list[threading.Thread]) -> None:
        if self.state is not SessionState.STARTING:
            raise RuntimeError(f"Cannot start session in {self.state} state")
        self.workers = workers
        started: list[threading.Thread] = []
        try:
            for worker in workers:
                worker.start()
                started.append(worker)
        except Exception:
            self.stop_event.set()
            self.frames.wake_all()
            for worker in started:
                worker.join(timeout=3.0)
            self.state = SessionState.ENDED
            self.clear_transient_results()
            raise
        self.state = SessionState.ACTIVE

    def stop(self, timeout: float = 3.0) -> None:
        if self.state is SessionState.ENDED:
            return
        self.state = SessionState.STOPPING
        self.stop_event.set()
        self.frames.wake_all()
        for worker in self.workers:
            if worker.ident is not None:
                worker.join(timeout=timeout)
        self.clear_transient_results()
        self.state = SessionState.ENDED

    def clear_transient_results(self) -> None:
        self.results.clear()
        self.frames.clear()
