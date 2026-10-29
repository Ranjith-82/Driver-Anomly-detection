from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


TALKING_ALERT_THRESHOLD = 4
TALKING_ALERT_COOLDOWN = 120.0
EMOJI_FONT_PATHS = (
    Path(r"C:\Windows\Fonts\seguiemj.ttf"),
    Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf"),
)

ALERT_CONTENT = {
    "FATIGUE": (
        "☕😴", "FATIGUE ALERT",
        "You may be feeling drowsy. Stop safely and rest.",
        "ZZ", (40, 95, 235),
    ),
    "DISTRACTED": (
        "⚠️👀", "DISTRACTED",
        "Stay focused on the road.",
        "!!", (35, 155, 245),
    ),
    "YAWNING": (
        "🥱", "YAWNING",
        "You may be getting tired. Take a break when safe.",
        "YAWN", (40, 175, 245),
    ),
    "TALKING": (
        "🗣️", "TALKING",
        "Please keep your focus on driving.",
        "TALK", (40, 175, 245),
    ),
    "PHONE": (
        "📵", "PHONE DETECTED",
        "Keep your phone away while driving.",
        "PHONE", (40, 95, 235),
    ),
    "CIGARETTE": (
        "🚭", "CIGARETTE DETECTED",
        "Keep attention on the road.",
        "SMOKE", (40, 95, 235),
    ),
    "CUP": (
        "🥤", "CUP DETECTED",
        "Keep attention on the road.",
        "CUP", (40, 175, 245),
    ),
    "BOTTLE": (
        "🍼", "BOTTLE DETECTED",
        "Keep attention on the road.",
        "BOTTLE", (40, 175, 245),
    ),
    "NORMAL": (
        "😊", "NORMAL",
        "Drive safely!",
        "OK", (80, 190, 115),
    ),
    "UNCERTAIN": (
        "❓", "STATE UNCERTAIN",
        "Driver state is uncertain. Please stay attentive.",
        "?", (40, 190, 235),
    ),
}


class TalkingAlertTracker:
    def __init__(self) -> None:
        self.episode_count = 0
        self.was_talking = False
        self.last_announcement_at = -float("inf")
        self.announcement_due = False

    def update(self, behavior: str, now: float) -> bool:
        self.announcement_due = False
        is_talking = behavior == "TALKING"
        if is_talking and not self.was_talking:
            self.episode_count += 1
            if (
                self.episode_count >= TALKING_ALERT_THRESHOLD
                and now - self.last_announcement_at >= TALKING_ALERT_COOLDOWN
            ):
                self.last_announcement_at = now
                self.announcement_due = True
        self.was_talking = is_talking
        return is_talking and self.episode_count >= TALKING_ALERT_THRESHOLD


def _content_key(behavior: str, alert_status: str, talking_alert_active: bool,
                 face_present: bool) -> str:
    status = alert_status.removesuffix(" (cooldown)").upper()
    if status != "NONE":
        if "FATIGUE" in status:
            return "FATIGUE"
        if "DISTRACT" in status:
            return "DISTRACTED"
        if "YAWN" in status:
            return "YAWNING"
        if "PHONE" in status:
            return "PHONE"
        if "CIGARETTE" in status:
            return "CIGARETTE"
        if "CUP" in status:
            return "CUP"
        if "BOTTLE" in status:
            return "BOTTLE"
    if not face_present:
        return "UNCERTAIN"
    if behavior in {"FATIGUE ALERT", "DISTRACTED", "YAWNING"}:
        return behavior.removesuffix(" ALERT")
    if behavior == "TALKING":
        return "TALKING" if talking_alert_active else "TALKING_MONITORING"
    if behavior == "NORMAL":
        return "NORMAL"
    return "UNCERTAIN"


@lru_cache(maxsize=8)
def _load_emoji_font(size: int):
    if ImageFont is None:
        return None
    for path in EMOJI_FONT_PATHS:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return None


def _draw_emoji(panel: np.ndarray, emoji: str, size: int) -> bool:
    font = _load_emoji_font(size)
    if font is None or Image is None or ImageDraw is None:
        return False
    image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    try:
        bounds = draw.textbbox((0, 0), emoji, font=font)
        x = max(0, (panel.shape[1] - (bounds[2] - bounds[0])) // 2)
        y = 112
        draw.text(
            (x - bounds[0], y - bounds[1]),
            emoji,
            font=font,
            embedded_color=True,
        )
    except (OSError, TypeError, ValueError):
        return False
    panel[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return True


def _wrap_text(text: str, width: int, scale: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0] > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_alert_panel(
    width: int,
    height: int,
    behavior: str,
    alert_status: str = "NONE",
    talking_alert_active: bool = False,
    talking_episode_count: int = 0,
    face_present: bool = True,
) -> np.ndarray:
    key = _content_key(
        behavior, alert_status, talking_alert_active, face_present
    )
    if key == "TALKING_MONITORING":
        emoji, title = "", "TALKING MONITORING"
        message = (
            f"Talking episodes: {talking_episode_count}/"
            f"{TALKING_ALERT_THRESHOLD}. Alert on episode "
            f"{TALKING_ALERT_THRESHOLD}."
        )
        fallback, accent = "", (40, 175, 245)
    else:
        emoji, title, message, fallback, accent = ALERT_CONTENT[key]
        if key == "TALKING":
            title = "REPEATED TALKING ALERT"
            message = (
                "Talking has been detected several times. "
                "Please focus on driving."
            )

    panel = np.full((height, width, 3), (24, 27, 34), dtype=np.uint8)
    cv2.rectangle(panel, (1, 1), (width - 2, height - 2), accent, 2)
    header = "LIVE DRIVER ALERT"
    cv2.putText(
        panel, header, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (225, 230, 238), 1, cv2.LINE_AA,
    )
    cv2.line(panel, (16, 48), (width - 16, 48), (70, 76, 86), 1)

    rendered = bool(emoji) and _draw_emoji(panel, emoji, min(96, height // 5))
    if emoji and not rendered:
        size = cv2.getTextSize(
            fallback, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )[0]
        cv2.putText(
            panel, fallback, (max(12, (width - size[0]) // 2), 170),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 2, cv2.LINE_AA,
        )

    title_scale = min(0.58, (width - 24) / max(
        cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0][0], 1
    ) * 0.58)
    title_width = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 2
    )[0][0]
    cv2.putText(
        panel, title, (max(12, (width - title_width) // 2), 300),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, accent, 2, cv2.LINE_AA,
    )
    lines = _wrap_text(message, width - 32, 0.46)
    for index, line in enumerate(lines[:5]):
        cv2.putText(
            panel, line, (16, 350 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (225, 230, 238), 1, cv2.LINE_AA,
        )
    return panel
