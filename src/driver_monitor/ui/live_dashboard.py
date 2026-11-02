from __future__ import annotations

from collections.abc import Mapping


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
