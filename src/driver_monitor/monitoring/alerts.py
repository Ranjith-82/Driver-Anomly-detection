from __future__ import annotations

from collections.abc import Mapping
import json
import os
import queue
import threading
from collections import deque
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from driver_monitor.profiles.repository import DriverProfileRepository


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


def notification_values(values: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("emotion")
        and key not in {"face_bounds", "face_crop"}
    }


def build_safety_alerts(values: Mapping[str, object]) -> list[str]:
    alerts: list[str] = []
    if values.get("face_present") and (
        values.get("behavior") == "FATIGUE ALERT"
        or values.get("fatigue_alert") is True
    ):
        alerts.append("FATIGUE ALERT")
    if values.get("gaze_distraction_confirmed") is True:
        alerts.append("DRIVER DISTRACTION DETECTED")
    elif values.get("behavior") == "DISTRACTED":
        alerts.append("DRIVER DISTRACTED")
    if values.get("yawn_event") is True:
        alerts.append("YAWNING DETECTED")
    detections = values.get("object_detections")
    if isinstance(detections, list):
        class_ids = sorted({
            item["class_id"]
            for item in detections
            if isinstance(item, dict)
            and item.get("class_id") in OBJECT_ALERTS
            and item.get("class_id") != 0
        })
        alerts.extend(OBJECT_ALERTS[class_id] for class_id in class_ids)
    if (
        values.get("driver_phone_alert") is True
        and values.get("driver_confirmed") is True
        and values.get("phone_detection_confirmed") is True
    ):
        alerts.append(OBJECT_ALERTS[0])
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
        self.database_path = database_path
        self.repository = DriverProfileRepository(database_path)
        self.telegram_source = telegram_source
        self.telegram_import_error = telegram_import_error
        self.bot_token = os.getenv("DRIVER_MONITOR_BOT_TOKEN", "")
        self.chat_id = os.getenv("DRIVER_MONITOR_CHAT_ID", "")

    def submit(self, reason, values, frame) -> None:
        item = (reason, notification_values(values), frame.copy())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            print("[TELEGRAM] Alert queue full; dropping stale alert")

    def submit_camera_error(self, error: str) -> None:
        item = ("CAMERA ERROR", {"camera_error": error}, None)
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
                reason, values, frame = item
                if reason == "CAMERA ERROR":
                    self.send_camera_error(str(values["camera_error"]))
                else:
                    self.send_alert(reason, values, frame)
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

    def send_alert(self, reason, values, frame) -> None:
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
        driver_name = values["driver_name"]
        driver_details = {}
        if driver_name not in {"UNKNOWN", "IDENTIFYING..."}:
            driver_details = next(
                (
                    profile.to_legacy_mapping()
                    for profile in self.repository.list_profiles()
                    if profile.full_name == driver_name
                ),
                {},
            )
        if not driver_details:
            driver_details = {"full_name": "Unknown driver"}
        details = (
            f'{reason}; EAR={values["average_ear"]}; MAR={values["mar"]}; '
            f'head={values["head_position"]}; pitch={values["pitch"]}; '
            f'yaw={values["yaw"]}'
        )
        sent = self.telegram_source.send_telegram_alert(
            driver_details, details, frame
        )
        status = "Sent" if sent else "Failed"
        print(
            f'[TELEGRAM] {status}: {reason} for {values["driver_name"]}'
        )

    def close(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.queue.put(None)
            self.join(timeout=8.0)


class AlertManager:
    """Preserves existing priority, event windows, and notification cooldowns."""

    def __init__(
        self,
        telegram_worker,
        cooldown: float = 60.0,
        distraction_delay: float = 0.5,
        yawn_cooldown: float = 10.0,
        fatigue_events: int = 3,
        fatigue_window: float = 60.0,
    ) -> None:
        self.telegram_worker = telegram_worker
        self.cooldown = cooldown
        self.distraction_delay = distraction_delay
        self.yawn_cooldown = yawn_cooldown
        self.fatigue_events_required = fatigue_events
        self.fatigue_window = fatigue_window
        self.fatigue_event_times = deque()
        self.fatigue_active = False
        self.fatigue_notification_sent = False
        self.last_sent = {}
        self.distraction_started = None
        self.gaze_distraction_alerted = False
        self.object_classes_active = set()
        self.pending_object_alerts = deque()
        self.last_alert = "NONE"

    def update(self, values, frame, now):
        behavior = values["behavior"]
        detected_object_classes = {
            detection["class_id"]
            for detection in values.get("object_detections", [])
            if detection.get("class_id") in OBJECT_ALERT_CONFIG
            and detection.get("class_id") != 0
        }
        if (
            values.get("driver_phone_alert") is True
            and values.get("driver_confirmed") is True
            and values.get("phone_detection_confirmed") is True
        ):
            detected_object_classes.add(0)
        new_object_classes = detected_object_classes - self.object_classes_active
        for class_id in sorted(new_object_classes):
            alert = OBJECT_ALERT_CONFIG[class_id]
            if class_id == 0:
                self.pending_object_alerts.appendleft(alert)
            else:
                self.pending_object_alerts.append(alert)
        self.object_classes_active = detected_object_classes

        fatigue_detected = (
            values["face_present"]
            and (behavior == "FATIGUE ALERT" or values["fatigue_alert"])
        )
        if fatigue_detected and not self.fatigue_active:
            self.fatigue_event_times.append(now)
            self.fatigue_active = True
            print(
                f"[ALERT] Fatigue event detected at {now:.2f}. "
                f"Total events: {len(self.fatigue_event_times)}"
            )
        elif not fatigue_detected:
            self.fatigue_active = False

        fatigue_cutoff = now - self.fatigue_window
        while (
            self.fatigue_event_times
            and self.fatigue_event_times[0] < fatigue_cutoff
        ):
            self.fatigue_event_times.popleft()
        if not self.fatigue_event_times:
            self.fatigue_notification_sent = False

        if behavior == "DISTRACTED":
            if self.distraction_started is None:
                self.distraction_started = now
        else:
            self.distraction_started = None
        distracted_duration = (
            0.0
            if self.distraction_started is None
            else now - self.distraction_started
        )

        gaze_distraction = values.get("gaze_distraction_confirmed", False)
        if not gaze_distraction:
            self.gaze_distraction_alerted = False

        reason = None
        key = None
        if self.pending_object_alerts:
            reason, key = self.pending_object_alerts.popleft()
        elif (
            len(self.fatigue_event_times) >= self.fatigue_events_required
            and not self.fatigue_notification_sent
        ):
            reason, key = "FATIGUE ALERT", "fatigue"
        elif gaze_distraction and not self.gaze_distraction_alerted:
            reason = (
                "WARNING: DRIVER DISTRACTION DETECTED. "
                "PLEASE LOOK AT THE ROAD."
            )
            key = "gaze_distraction"
            self.gaze_distraction_alerted = True
        elif (
            behavior == "DISTRACTED"
            and distracted_duration >= self.distraction_delay
        ):
            reason, key = "DRIVER DISTRACTED", "distraction"
        elif values.get("yawn_event", False):
            reason, key = "YAWNING DETECTED", "yawn"

        if reason is None:
            self.last_alert = "NONE"
        elif now - self.last_sent.get(key, -float("inf")) >= (
            self.yawn_cooldown if key == "yawn" else self.cooldown
        ):
            self.last_sent[key] = now
            self.last_alert = reason
            if key == "fatigue":
                self.fatigue_notification_sent = True
            if reason == "BOTTLE DETECTED":
                print(f"[ALERT] Local alert only; Telegram disabled: {reason}")
            else:
                self.telegram_worker.submit(
                    reason, notification_values(values), frame
                )
                print(f"[ALERT] Alert submitted: {reason}")
        else:
            self.last_alert = f"{reason} (cooldown)"
        return distracted_duration
