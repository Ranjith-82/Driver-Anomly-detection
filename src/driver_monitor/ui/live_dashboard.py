from __future__ import annotations

from collections.abc import Mapping


def emotion_display(values: Mapping[str, object]) -> tuple[str, str]:
    status = str(values.get("emotion_status", "uncertain")).casefold()
    label = str(values.get("emotion_label", "Uncertain"))
    if status == "ready":
        return label, "ready"
    if status == "no_face":
        return "No face", "no_face"
    if status == "unavailable":
        return "Uncertain", "unavailable"
    if status == "stale":
        return "Uncertain", "stale"
    return "Uncertain", "uncertain"
