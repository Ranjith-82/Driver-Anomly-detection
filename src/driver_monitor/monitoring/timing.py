from __future__ import annotations

from datetime import datetime
import time


_PROCESS_STARTED_AT = time.monotonic()


def log_timing(stage: str, **details: object) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    elapsed = time.monotonic() - _PROCESS_STARTED_AT
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" {fields}" if fields else ""
    print(f"[TIMING] {timestamp} elapsed={elapsed:.3f}s {stage}{suffix}")
