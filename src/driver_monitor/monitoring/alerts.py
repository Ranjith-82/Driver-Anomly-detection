from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
import queue
import threading
import uuid
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OBJECT_ALERTS = {
    0: "POSSIBLE DRIVER PHONE USE",
    1: "CIGARETTE DETECTED",
    2: "CUP DETECTED",
    3: "BOTTLE DETECTED",
}
OBJECT_ALERT_CONFIG = {
    0: ("POSSIBLE DRIVER PHONE USE", "mobile_phone"),
    1: ("CIGARETTE DETECTED", "cigarette"),
    2: ("CUP DETECTED", "cup"),
    3: ("BOTTLE DETECTED", "bottle"),
}
OBJECT_ALERT_PRIORITY_KEYS = {
    0: "phone",
    1: "cigarette",
    2: "cup",
    3: "bottle",
}
ALERT_PRIORITY = (
    "phone",
    "bottle",
    "gaze_distraction",
    "distraction",
    "yawn",
    "talking",
    "cigarette",
    "cup",
)
ALERT_DEFINITIONS = {
    "FATIGUE": (
        "DRIVER FATIGUE ALERT",
        "FATIGUE DETECTED",
        "Driver fatigue or prolonged eye closure detected.",
    ),
    "YAWNING": (
        "YAWNING DETECTED",
        "YAWNING DETECTED",
        "Driver yawning detected. Possible fatigue indicator.",
    ),
    "DISTRACTION": (
        "DRIVER DISTRACTION ALERT",
        "DRIVER DISTRACTION DETECTED",
        "Driver attention diverted from the road.",
    ),
    "TALKING": (
        "TALKING DETECTED",
        "TALKING DETECTED",
        "Driver talking detected.",
    ),
    "PHONE_USE": (
        "POSSIBLE DRIVER PHONE USE",
        "POSSIBLE DRIVER PHONE USE",
        "A phone was detected in association with the driver.",
    ),
    "SMOKING": (
        "POSSIBLE SMOKING DETECTED",
        "POSSIBLE SMOKING DETECTED",
        "Cigarette detected near the driver.",
    ),
    "CUP_DETECTION": (
        "CUP DETECTED",
        "CUP DETECTED",
        "Cup detected in the monitoring area.",
    ),
}
@dataclass(frozen=True)
class AlertEvent:
    event_id: str
    alert_type: str
    source: str
    title: str
    screenshot_title: str
    message: str
    reason: str
    timestamp: str
    driver_name: str | None
    confirmed: bool
    detection_frame_id: int | None
    screenshot_frame_id: int | None
    detection_timestamp: float | None
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PendingAlert:
    reason: str
    key: str
    event: AlertEvent | None
    frame: object


def _finite_number(values: Mapping[str, object], key: str) -> float | None:
    raw_value = values.get(key)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    value = float(raw_value)
    return value if math.isfinite(value) else None


def _frame_id(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _confirmed_driver_name(values: Mapping[str, object]) -> str | None:
    name = values.get("driver_name")
    if (
        values.get("driver_confirmed") is not True
        or not isinstance(name, str)
        or name.strip().casefold() in {"", "unknown", "identifying..."}
    ):
        return None
    return name.strip()


def _confirmed_object(values: Mapping[str, object], class_id: int):
    detections = values.get("confirmed_object_detections")
    if not isinstance(detections, list):
        return None
    return next(
        (
            detection
            for detection in detections
            if isinstance(detection, dict)
            and detection.get("class_id") == class_id
            and detection.get("confirmed") is True
            and detection.get("current") is not False
        ),
        None,
    )


def _format_detail(label: str, value: object, suffix: str = "") -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if label == "Confirmed talking episodes":
            return f"{label}: {int(number)}"
        return f"{label}: {number:.3f}{suffix}"
    if isinstance(value, str) and value.strip():
        return f"{label}: {value.strip()}"
    return None


def _event_details(
    alert_type: str, values: Mapping[str, object]
) -> tuple[tuple[str, str], ...]:
    details: list[tuple[str, str]] = []

    def add(label: str, value: object, suffix: str = "") -> None:
        formatted = _format_detail(label, value, suffix)
        if formatted is not None:
            details.append((label, formatted.split(": ", 1)[1]))

    if alert_type == "FATIGUE":
        add("EAR", _finite_number(values, "average_ear"))
        add(
            "Eye closure duration",
            _finite_number(values, "eyes_closed_duration"),
            " s",
        )
        add("Head direction", values.get("head_position"))
    elif alert_type == "YAWNING":
        add("MAR", _finite_number(values, "mar"))
    elif alert_type == "DISTRACTION":
        add("Head direction", values.get("head_position"))
        add("Gaze direction", values.get("gaze_direction"))
        duration = _finite_number(values, "gaze_distraction_elapsed")
        if duration is None:
            duration = _finite_number(values, "behavior_distraction_duration")
        add("Duration", duration, " s")
    elif alert_type == "TALKING":
        add("Confirmed talking episodes", values.get("talking_episode_count"))
    else:
        class_id = {"PHONE_USE": 0, "SMOKING": 1, "CUP_DETECTION": 2}[
            alert_type
        ]
        detection = _confirmed_object(values, class_id)
        if detection is not None:
            add("Object", detection.get("name") or OBJECT_ALERTS[class_id])
            add("Detection confidence", detection.get("confidence"))
        add("Confirmation status", "Confirmed")
    return tuple(details)


def _build_event_message(event: AlertEvent) -> str:
    lines = [event.title, ""]
    if event.driver_name is not None:
        lines.append(f"Driver: {event.driver_name}")
    lines.extend([
        f"Event: {event.alert_type.replace('_', ' ').title()}",
        f"Reason: {event.reason}",
        f"Detail: {event.message}",
    ])
    lines.extend(f"{label}: {value}" for label, value in event.details)
    lines.append(f"Time: {event.timestamp}")
    lines.append(f"Event ID: {event.event_id}")
    return "\n".join(lines)


def _legacy_alert_type(reason: str) -> str:
    normalized = reason.casefold()
    if "fatigue" in normalized:
        return "FATIGUE"
    if "yawn" in normalized:
        return "YAWNING"
    if "talk" in normalized:
        return "TALKING"
    if "phone" in normalized:
        return "PHONE_USE"
    if "cigarette" in normalized or "smok" in normalized:
        return "SMOKING"
    if "cup" in normalized:
        return "CUP_DETECTION"
    if "distract" in normalized or "gaze" in normalized:
        return "DISTRACTION"
    raise ValueError(f"Unsupported legacy alert reason: {reason!r}")


def _create_alert_event(
    alert_type: str,
    source: str,
    values: Mapping[str, object],
    *,
    screenshot_frame_id: int | None,
    distracted_duration: float | None = None,
) -> AlertEvent:
    title, screenshot_title, message = ALERT_DEFINITIONS[alert_type]
    reason_by_type = {
        "FATIGUE": (
            "Prolonged eye closure was detected by the fatigue monitor."
        ),
        "YAWNING": (
            "Mouth opening crossed the configured yawning threshold and "
            "temporal validation confirmed the event."
        ),
        "DISTRACTION": (
            "Sustained non-forward gaze was confirmed."
            if source == "gaze_temporal"
            else "Sustained behavioural distraction was confirmed."
        ),
        "TALKING": (
            "Talking was confirmed by the existing temporal episode validation."
        ),
        "PHONE_USE": (
            "Phone tracking and driver association confirmed the detection."
        ),
        "SMOKING": (
            "Cigarette object tracking confirmed the detection in the "
            "monitoring area."
        ),
        "CUP_DETECTION": (
            "Cup object tracking confirmed the detection."
        ),
    }
    detection_frame_id = _frame_id(
        values,
        "object_detection_frame_id"
        if alert_type in {"PHONE_USE", "SMOKING", "CUP_DETECTION"}
        else "behavior_frame_id",
    )
    detection_timestamp = _finite_number(
        values,
        "object_detection_timestamp"
        if alert_type in {"PHONE_USE", "SMOKING", "CUP_DETECTION"}
        else "behavior_timestamp",
    )
    detail_values = dict(values)
    if distracted_duration is not None:
        detail_values["behavior_distraction_duration"] = distracted_duration
    event = AlertEvent(
        event_id=uuid.uuid4().hex,
        alert_type=alert_type,
        source=source,
        title=title,
        screenshot_title=screenshot_title,
        message=message,
        reason=reason_by_type[alert_type],
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        driver_name=_confirmed_driver_name(values),
        confirmed=True,
        detection_frame_id=detection_frame_id,
        screenshot_frame_id=screenshot_frame_id,
        detection_timestamp=detection_timestamp,
        details=_event_details(alert_type, detail_values),
    )
    return event


def notification_values(values: Mapping[str, object]) -> dict[str, object]:
    private_keys = {
        "phone",
        "phone_number",
        "dob",
        "date_of_birth",
        "blood_group",
        "address",
        "city",
        "driver_details",
    }
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("emotion")
        and key not in {"face_bounds", "face_crop"}
        and key.casefold() not in private_keys
    }


def _confirmed_object_class_ids(values: Mapping[str, object]) -> set[int]:
    detections = values.get("confirmed_object_detections")
    if not isinstance(detections, list):
        return set()
    return {
        item["class_id"]
        for item in detections
        if isinstance(item, dict)
        and item.get("confirmed") is True
        and item.get("class_id") in OBJECT_ALERTS
        and item.get("class_id") != 0
    }


def build_safety_alerts(values: Mapping[str, object]) -> list[str]:
    alerts: list[str] = []
    if values.get("face_present") and (
        values.get("behavior") == "FATIGUE ALERT"
        or values.get("fatigue_alert") is True
    ):
        alerts.append("FATIGUE ALERT")

    if (
        values.get("driver_phone_alert") is True
        and values.get("driver_confirmed") is True
        and values.get("phone_detection_confirmed") is True
    ):
        alerts.append(OBJECT_ALERTS[0])

    class_ids = _confirmed_object_class_ids(values)
    if 3 in class_ids:
        alerts.append(OBJECT_ALERTS[3])

    if values.get("gaze_distraction_confirmed") is True:
        alerts.append("DRIVER DISTRACTION DETECTED")
    if values.get("behavior") == "DISTRACTED":
        alerts.append("DRIVER DISTRACTED")
    if values.get("yawn_event") is True:
        alerts.append("YAWNING DETECTED")
    if 1 in class_ids:
        alerts.append(OBJECT_ALERTS[1])
    if 2 in class_ids:
        alerts.append(OBJECT_ALERTS[2])
    return alerts


class TelegramWorker(threading.Thread):
    """Sends sanitized alert snapshots away from monitoring workers."""

    def __init__(
        self,
        stop_event: threading.Event,
        database_path,
        cooldown: float,
        telegram_source=None,
        telegram_import_error: str | None = None,
    ) -> None:
        super().__init__(name="telegram-worker", daemon=True)
        self.stop_event = stop_event
        self.cooldown = cooldown
        self.queue: queue.Queue = queue.Queue(maxsize=8)
        self.telegram_source = telegram_source
        self.telegram_import_error = telegram_import_error
        self.bot_token = os.getenv("DRIVER_MONITOR_BOT_TOKEN", "")
        self.chat_id = os.getenv("DRIVER_MONITOR_CHAT_ID", "")

    def submit(self, reason, values, frame, *, event=None) -> None:
        item = (
            reason,
            notification_values(values),
            frame.copy(),
            event,
        )
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            print("[TELEGRAM] Alert queue full; dropping stale alert")

    def submit_camera_error(self, error: str) -> None:
        item = ("CAMERA ERROR", {"camera_error": error}, None, None)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self.queue.task_done()
                print("[TELEGRAM] Dropped a queued alert for camera error")
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                print(
                    "[TELEGRAM] Alert queue full; camera error notification "
                    "could not be queued"
                )

    def run(self) -> None:
        print("[TELEGRAM] Worker started")
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    break
                reason, values, frame, event = item
                if reason == "CAMERA ERROR":
                    self.send_camera_error(str(values["camera_error"]))
                else:
                    self.send_alert(reason, values, frame, event=event)
            except Exception as error:
                print(f"[TELEGRAM] ERROR: {error}")
            finally:
                self.queue.task_done()
        print("[TELEGRAM] Worker stopped")

    def send_camera_error(self, error: str) -> None:
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                "Telegram credentials are missing. Set DRIVER_MONITOR_BOT_TOKEN "
                "and DRIVER_MONITOR_CHAT_ID in this PowerShell window."
            )
        request = Request(
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
            data=urlencode({
                "chat_id": self.chat_id,
                "text": f"Driver monitor camera error: {error}",
            }).encode("utf-8"),
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError("Telegram rejected the camera error notification")
        print("[TELEGRAM] Sent camera error notification")

    def send_alert(self, reason, values, frame, *, event=None) -> None:
        if self.telegram_source is None:
            detail = (
                f": {self.telegram_import_error}"
                if self.telegram_import_error
                else ""
            )
            raise RuntimeError(
                f"Existing Telegram module could not be imported{detail}"
            )
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                "Telegram credentials are missing. Set DRIVER_MONITOR_BOT_TOKEN "
                "and DRIVER_MONITOR_CHAT_ID in this PowerShell window."
            )
        self.telegram_source.Config.BOT_TOKEN = self.bot_token
        self.telegram_source.Config.CHAT_ID = self.chat_id
        if event is None:
            event_type = _legacy_alert_type(str(reason))
            event = _create_alert_event(
                event_type,
                "legacy_alert",
                values,
                screenshot_frame_id=None,
            )
        driver_details = (
            {"full_name": event.driver_name}
            if event.driver_name is not None
            else {}
        )
        message = _build_event_message(event)
        event_values = dict(event.details)
        print(
            "[ALERT EVENT] "
            f"id={event.event_id} type={event.alert_type} "
            f"source={event.source} timestamp={event.timestamp} "
            f"values={event_values} confirmed={event.confirmed} "
            f"detection_timestamp={event.detection_timestamp} "
            f"detection_frame_id={event.detection_frame_id} "
            f"screenshot_frame_id={event.screenshot_frame_id}"
        )
        sent = self.telegram_source.send_telegram_alert(
            driver_details,
            message,
            frame,
            alert_type=event.alert_type,
            screenshot_title=event.screenshot_title,
            screenshot_reason=event.reason,
            screenshot_details=event.details,
            event_id=event.event_id,
            timestamp=event.timestamp,
            screenshot_frame_id=event.screenshot_frame_id,
        )
        status = "Sent" if sent else "Failed"
        print(f"[TELEGRAM] {status}: event={event.event_id} type={event.alert_type}")

    def close(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.queue.put(None)
            self.join(timeout=8.0)


class AlertManager:
    """Schedules alerts by priority while preserving event windows and cooldowns."""

    def __init__(
        self,
        telegram_worker,
        cooldown: float = 60.0,
        distraction_delay: float = 0.5,
        yawn_cooldown: float = 10.0,
        talking_cooldown: float = 120.0,
    ) -> None:
        self.telegram_worker = telegram_worker
        self.cooldown = cooldown
        self.distraction_delay = distraction_delay
        self.yawn_cooldown = yawn_cooldown
        self.talking_cooldown = talking_cooldown
        self.fatigue_active = False
        self.last_sent = {}
        self.distraction_started = None
        self.gaze_distraction_alerted = False
        self.behavior_distraction_alerted = False
        self.yawn_alerted = False
        self.talking_alerted = False
        self.object_classes_active = set()
        self.pending_alerts = {}
        self.last_alert = "NONE"

    def update(
        self,
        values,
        frame,
        now,
        *,
        frame_id: int | None = None,
        frame_history: Mapping[int, object] | None = None,
    ):
        behavior = values["behavior"]
        frame_history = frame_history or {}

        def enqueue(
            priority_key: str,
            reason: str,
            cooldown_key: str,
            alert_type: str,
            source: str,
            *,
            event_values: Mapping[str, object] = values,
            distraction_duration: float = 0.0,
        ) -> None:
            if priority_key in self.pending_alerts:
                return
            detection_frame_key = (
                "object_detection_frame_id"
                if alert_type in {"PHONE_USE", "SMOKING", "CUP_DETECTION"}
                else "behavior_frame_id"
            )
            detection_frame_id = _frame_id(
                event_values, detection_frame_key
            )
            screenshot_frame_id = (
                detection_frame_id
                if detection_frame_id in frame_history
                else frame_id
            )
            screenshot = frame_history.get(screenshot_frame_id, frame)
            event = _create_alert_event(
                alert_type,
                source,
                event_values,
                screenshot_frame_id=screenshot_frame_id,
                distracted_duration=distraction_duration,
            )
            self.pending_alerts[priority_key] = _PendingAlert(
                reason, cooldown_key, event, screenshot
            )

        detected_object_classes = _confirmed_object_class_ids(values)
        if (
            values.get("driver_phone_alert") is True
            and values.get("driver_confirmed") is True
            and values.get("phone_detection_confirmed") is True
        ):
            detected_object_classes.add(0)
        new_object_classes = detected_object_classes - self.object_classes_active
        for class_id in sorted(new_object_classes):
            reason, key = OBJECT_ALERT_CONFIG[class_id]
            priority_key = OBJECT_ALERT_PRIORITY_KEYS[class_id]
            if class_id == 0:
                alert_type, source = "PHONE_USE", "phone_validator"
            elif class_id == 1:
                alert_type, source = "SMOKING", "object_validator"
            elif class_id == 2:
                alert_type, source = "CUP_DETECTION", "object_validator"
            else:
                alert_type = None
                source = ""
            if alert_type is None:
                self.pending_alerts.setdefault(
                    priority_key, _PendingAlert(reason, key, None, frame)
                )
            else:
                enqueue(
                    priority_key,
                    reason,
                    key,
                    alert_type,
                    source,
                )
        self.object_classes_active = detected_object_classes

        fatigue_detected = (
            values.get("face_present") is True
            and (
                behavior == "FATIGUE ALERT"
                or values.get("fatigue_alert") is True
            )
        )
        if fatigue_detected and not self.fatigue_active:
            print(
                f"[ALERT] Fatigue detected at {now:.2f}; "
                "Telegram notification is eligible immediately"
            )
        self.fatigue_active = fatigue_detected

        gaze_distraction = values.get("gaze_distraction_confirmed") is True
        if behavior == "DISTRACTED":
            if self.distraction_started is None:
                self.distraction_started = now
        else:
            self.distraction_started = None
            self.behavior_distraction_alerted = False
        distracted_duration = (
            0.0
            if self.distraction_started is None
            else now - self.distraction_started
        )

        if not gaze_distraction:
            self.gaze_distraction_alerted = False
        elif not self.gaze_distraction_alerted:
            enqueue(
                "gaze_distraction",
                (
                    "WARNING: DRIVER DISTRACTION DETECTED. "
                    "PLEASE LOOK AT THE ROAD."
                ),
                "gaze_distraction",
                "DISTRACTION",
                "gaze_temporal",
                distraction_duration=_finite_number(
                    values, "gaze_distraction_elapsed"
                ) or 0.0,
            )
            self.gaze_distraction_alerted = True

        if (
            behavior == "DISTRACTED"
            and distracted_duration >= self.distraction_delay
            and not self.behavior_distraction_alerted
            and not gaze_distraction
        ):
            enqueue(
                "distraction",
                "DRIVER DISTRACTED",
                "distraction",
                "DISTRACTION",
                "behavior_temporal",
                distraction_duration=distracted_duration,
            )
            self.behavior_distraction_alerted = True
        elif (
            behavior == "DISTRACTED"
            and distracted_duration >= self.distraction_delay
            and gaze_distraction
        ):
            self.behavior_distraction_alerted = True

        yawn_event = values.get("yawn_event") is True
        if not yawn_event:
            self.yawn_alerted = False
        elif not self.yawn_alerted:
            enqueue(
                "yawn",
                "YAWNING DETECTED",
                "yawn",
                "YAWNING",
                "mouth_temporal",
            )
            self.yawn_alerted = True

        talking_active = (
            values.get("talking_alert_active") is True
            and values.get("face_present") is True
        )
        if not talking_active:
            self.talking_alerted = False
        elif not self.talking_alerted:
            enqueue(
                "talking",
                "TALKING DETECTED",
                "talking",
                "TALKING",
                "talking_temporal",
            )
            self.talking_alerted = True

        reason = None
        key = None
        pending_alert = None
        if (
            fatigue_detected
            and now - self.last_sent.get("fatigue", -float("inf"))
            >= self.cooldown
        ):
            event = _create_alert_event(
                "FATIGUE",
                "eye_closure_monitor",
                values,
                screenshot_frame_id=(
                    _frame_id(values, "behavior_frame_id")
                    if _frame_id(values, "behavior_frame_id") in frame_history
                    else frame_id
                ),
            )
            screenshot_frame_id = event.screenshot_frame_id
            pending_alert = _PendingAlert(
                "FATIGUE ALERT",
                "fatigue",
                event,
                frame_history.get(screenshot_frame_id, frame),
            )
        else:
            for priority_key in ALERT_PRIORITY:
                pending = self.pending_alerts.get(priority_key)
                if pending is None:
                    continue
                candidate_key = pending.key
                candidate_cooldown = (
                    self.yawn_cooldown
                    if candidate_key == "yawn"
                    else self.talking_cooldown
                    if candidate_key == "talking"
                    else self.cooldown
                )
                if (
                    now - self.last_sent.get(candidate_key, -float("inf"))
                    >= candidate_cooldown
                ):
                    pending_alert = pending
                    reason, key = pending.reason, pending.key
                    del self.pending_alerts[priority_key]
                    break

        if pending_alert is None:
            pending = next(
                (
                    self.pending_alerts[priority_key]
                    for priority_key in ALERT_PRIORITY
                    if priority_key in self.pending_alerts
                ),
                None,
            )
            self.last_alert = (
                f"{pending.reason} (cooldown)" if pending else "NONE"
            )
        else:
            reason, key = pending_alert.reason, pending_alert.key
            self.last_sent[key] = now
            self.last_alert = reason
            if reason == "BOTTLE DETECTED":
                print(f"[ALERT] Local alert only; Telegram disabled: {reason}")
            else:
                event = pending_alert.event
                if event is None:
                    raise RuntimeError(
                        f"Telegram alert {reason!r} has no event context"
                    )
                payload = {
                    "alert_type": event.alert_type,
                    "source": event.source,
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "driver_name": event.driver_name,
                    "driver_confirmed": event.confirmed,
                    "detection_frame_id": event.detection_frame_id,
                    "screenshot_frame_id": event.screenshot_frame_id,
                    **dict(event.details),
                }
                self.telegram_worker.submit(
                    reason,
                    payload,
                    pending_alert.frame,
                    event=event,
                )
                print(f"[ALERT] Alert submitted: {reason}")
        return distracted_duration
