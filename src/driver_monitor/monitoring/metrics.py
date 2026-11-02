from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    target_seconds: float = 2.0
    max_samples: int = 1800
    _last_signature: Hashable | None = None
    _latencies: list[float] = field(default_factory=list)
    _within_target: int = 0

    def __post_init__(self) -> None:
        if self.target_seconds <= 0 or self.max_samples <= 0:
            raise ValueError("Latency target and sample limit must be positive")

    def record(
        self,
        captured_at: float,
        rendered_at: float,
        signature: Hashable,
    ) -> bool:
        if not math.isfinite(captured_at) or not math.isfinite(rendered_at):
            raise ValueError("Capture and render timestamps must be finite")
        if rendered_at < captured_at:
            raise ValueError("Render timestamp cannot precede capture timestamp")
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        latency = rendered_at - captured_at
        if len(self._latencies) == self.max_samples:
            removed = self._latencies.pop(0)
            if removed <= self.target_seconds:
                self._within_target -= 1
        self._latencies.append(latency)
        if latency <= self.target_seconds:
            self._within_target += 1
        return True

    def summary(self) -> dict[str, float | int]:
        if not self._latencies:
            return {
                "samples": 0,
                "within_target": 0,
                "fraction_within_target": 0.0,
                "p50_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
            }
        ordered = sorted(self._latencies)

        def percentile(fraction: float) -> float:
            index = max(0, math.ceil(fraction * len(ordered)) - 1)
            return ordered[index]

        return {
            "samples": len(ordered),
            "within_target": self._within_target,
            "fraction_within_target": self._within_target / len(ordered),
            "p50_seconds": percentile(0.50),
            "p95_seconds": percentile(0.95),
            "max_seconds": ordered[-1],
        }

    def clear(self) -> None:
        self._last_signature = None
        self._latencies.clear()
        self._within_target = 0
