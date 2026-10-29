from __future__ import annotations

from typing import Any


PHONE_CLASS_ID = 0
PHONE_TRACK_IOU_THRESHOLD = 0.1


def _intersection_over_union(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _valid_phone_detection(detection: Any) -> bool:
    if (
        not isinstance(detection, dict)
        or detection.get("class_id") != PHONE_CLASS_ID
    ):
        return False
    box = detection.get("box")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (int(value) for value in box)
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1 and 0.0 <= confidence <= 1.0


def _near_identified_driver(
    phone_box: tuple[int, ...],
    driver_location: tuple[int, ...] | list[int] | None,
) -> bool:
    if driver_location is None or len(driver_location) != 4:
        return False
    top, right, bottom, left = (int(value) for value in driver_location)
    face_width = max(1, right - left)
    face_height = max(1, bottom - top)
    phone_center_x = (phone_box[0] + phone_box[2]) / 2
    phone_center_y = (phone_box[1] + phone_box[3]) / 2
    return (
        left - 0.35 * face_width <= phone_center_x <= right + 0.35 * face_width
        and top - 0.25 * face_height
        <= phone_center_y
        <= bottom + 1.25 * face_height
    )


class PhoneDetectionValidator:
    """Tracks persistent phone predictions and a separate driver-use suspicion."""

    def __init__(
        self,
        *,
        persistence_seconds: float = 0.3,
        use_persistence_seconds: float = 0.75,
        alert_persistence_seconds: float = 0.5,
        max_missed_frames: int = 2,
        stale_after_seconds: float = 1.0,
    ) -> None:
        if (
            persistence_seconds < 0
            or use_persistence_seconds < 0
            or alert_persistence_seconds < 0
        ):
            raise ValueError("Phone persistence durations cannot be negative")
        if max_missed_frames < 0 or stale_after_seconds <= 0:
            raise ValueError("Phone track expiry settings are invalid")
        self.persistence_seconds = persistence_seconds
        self.use_persistence_seconds = use_persistence_seconds
        self.alert_persistence_seconds = alert_persistence_seconds
        self.max_missed_frames = max_missed_frames
        self.stale_after_seconds = stale_after_seconds
        self.frame_index = 0
        self._next_track_id = 1
        self._tracks: list[dict[str, Any]] = []

    def update(
        self,
        detections: list[dict[str, Any]],
        timestamp: float,
        *,
        driver_confirmed: bool = False,
        driver_location: tuple[int, ...] | list[int] | None = None,
    ) -> dict[str, Any]:
        self.frame_index += 1
        phones = [
            dict(detection)
            for detection in detections
            if _valid_phone_detection(detection)
        ]
        unmatched_tracks = set(range(len(self._tracks)))
        matched: list[tuple[int, dict[str, Any]]] = []

        for detection in phones:
            box = tuple(int(value) for value in detection["box"])
            best_index = None
            best_iou = PHONE_TRACK_IOU_THRESHOLD
            for track_index in unmatched_tracks:
                score = _intersection_over_union(
                    self._tracks[track_index]["box"], box)
                if score > best_iou:
                    best_index = track_index
                    best_iou = score
            if best_index is None:
                track = {
                    "id": self._next_track_id,
                    "box": box,
                    "last_detection": detection,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "last_frame": self.frame_index,
                    "hits": 1,
                    "confidence": float(detection["confidence"]),
                    "confirmed": False,
                    "driver_use_started": None,
                    "driver_use_last_seen": None,
                    "driver_alert_started": None,
                }
                self._next_track_id += 1
                self._tracks.append(track)
                track_index = len(self._tracks) - 1
            else:
                track_index = best_index
                unmatched_tracks.remove(track_index)
                track = self._tracks[track_index]
                track["box"] = box
                track["last_detection"] = detection
                track["last_seen"] = timestamp
                track["last_frame"] = self.frame_index
                track["hits"] += 1
                track["confidence"] = float(detection["confidence"])
            if (
                track["hits"] >= 2
                and timestamp - track["first_seen"] >= self.persistence_seconds
            ):
                track["confirmed"] = True
            matched.append((track["id"], detection))

        self._tracks = [
            track
            for track in self._tracks
            if self.frame_index - track["last_frame"] <= self.max_missed_frames
            and timestamp - track["last_seen"] <= self.stale_after_seconds
        ]
        live_track_ids = {track["id"] for track in self._tracks}
        current_detections = {
            track_id: detection
            for track_id, detection in matched
            if track_id in live_track_ids
        }

        confirmed_detections = []
        suspected_driver_use = False
        driver_phone_alert = False
        for track in self._tracks:
            if not track["confirmed"]:
                continue
            detection = current_detections.get(track["id"])
            if detection is not None:
                output = dict(detection)
                output["track_id"] = track["id"]
                output["confirmed"] = True
                output["current"] = True
                confirmed_detections.append(output)
            else:
                output = dict(track["last_detection"])
                output["track_id"] = track["id"]
                output["confirmed"] = True
                output["current"] = False
                confirmed_detections.append(output)

            near_driver = (
                driver_confirmed
                and detection is not None
                and _near_identified_driver(track["box"], driver_location)
            )
            if near_driver:
                if track["driver_use_started"] is None:
                    track["driver_use_started"] = timestamp
                track["driver_use_last_seen"] = timestamp
            elif detection is not None:
                track["driver_use_started"] = None
                track["driver_use_last_seen"] = None
                track["driver_alert_started"] = None
            track_suspected = (
                track["driver_use_started"] is not None
                and track["driver_use_last_seen"] is not None
                and timestamp - track["driver_use_started"]
                >= self.use_persistence_seconds
                and timestamp - track["driver_use_last_seen"]
                <= self.stale_after_seconds
            )
            if track_suspected and track["driver_alert_started"] is None:
                track["driver_alert_started"] = timestamp
            if not track_suspected:
                track["driver_alert_started"] = None
            suspected_driver_use = suspected_driver_use or track_suspected
            driver_phone_alert = driver_phone_alert or (
                track_suspected
                and track["driver_alert_started"] is not None
                and timestamp - track["driver_alert_started"]
                >= self.alert_persistence_seconds
            )

        return {
            "phone_present": bool(phones),
            "phone_detection_confirmed": bool(confirmed_detections),
            "confirmed_phone_detections": confirmed_detections,
            "suspected_driver_phone_use": suspected_driver_use,
            "driver_phone_alert": driver_phone_alert,
            "phone_track_state": (
                "driver_phone_alert"
                if driver_phone_alert
                else "suspected_driver_use"
                if suspected_driver_use
                else "confirmed"
                if confirmed_detections
                else "candidate"
                if phones
                else "none"
            ),
        }


class BehaviorTemporalFilter:
    """Debounces transient behavior labels while keeping fatigue immediate."""

    def __init__(
        self,
        *,
        talking_seconds: float = 0.35,
        yawning_seconds: float = 0.15,
        distraction_seconds: float = 0.1,
        clear_seconds: float = 0.15,
    ) -> None:
        durations = (
            talking_seconds,
            yawning_seconds,
            distraction_seconds,
            clear_seconds,
        )
        if any(duration < 0 for duration in durations):
            raise ValueError("Behavior persistence durations cannot be negative")
        self.durations = {
            "TALKING": talking_seconds,
            "YAWNING": yawning_seconds,
            "DISTRACTED": distraction_seconds,
        }
        self.clear_seconds = clear_seconds
        self.current = "NORMAL"
        self._candidate: str | None = None
        self._candidate_since: float | None = None

    def reset(self, behavior: str = "FACE NOT DETECTED") -> str:
        self.current = behavior
        self._candidate = None
        self._candidate_since = None
        return behavior

    def update(self, behavior: str, timestamp: float) -> str:
        if behavior in {"FATIGUE ALERT", "FACE NOT DETECTED", "CALIBRATING"}:
            return self.reset(behavior)
        if behavior == self.current:
            self._candidate = None
            self._candidate_since = None
            return self.current
        if behavior != self._candidate:
            self._candidate = behavior
            self._candidate_since = timestamp
        required = (
            self.durations.get(behavior, self.clear_seconds)
            if behavior != "NORMAL"
            else self.clear_seconds
        )
        if timestamp - self._candidate_since >= required:
            self.current = behavior
            self._candidate = None
            self._candidate_since = None
        return self.current
