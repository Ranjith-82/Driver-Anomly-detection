from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from driver_monitor.monitoring.alerts import AlertManager, TelegramWorker
from driver_monitor.monitoring.legacy_runtime import telegram_source


class AlertEventTests(unittest.TestCase):
    def base_values(self):
        return {
            "behavior": "NORMAL",
            "face_present": True,
            "fatigue_alert": False,
            "driver_confirmed": True,
            "driver_name": "Test Driver",
            "behavior_frame_id": 7,
            "behavior_timestamp": 123.0,
            "object_detection_frame_id": 7,
            "object_detection_timestamp": 123.0,
            "confirmed_object_detections": [],
        }

    def run_alert(self, values):
        worker = Mock()
        manager = AlertManager(worker, distraction_delay=0.0)
        manager.update(
            values,
            np.zeros((40, 60, 3), dtype=np.uint8),
            now=10.0,
            frame_id=8,
            frame_history={
                7: np.full((40, 60, 3), 127, dtype=np.uint8)
            },
        )
        self.assertTrue(worker.submit.called)
        return worker.submit.call_args

    def test_each_supported_event_keeps_its_type_reason_and_source_frame(self):
        phone = {
            "class_id": 0,
            "name": "Phone",
            "confidence": 0.87,
            "confirmed": True,
        }
        cigarette = {
            "class_id": 1,
            "name": "Cigarette",
            "confidence": 0.76,
            "confirmed": True,
        }
        cup = {
            "class_id": 2,
            "name": "Cup",
            "confidence": 0.81,
            "confirmed": True,
        }
        cases = (
            (
                "FATIGUE",
                "FATIGUE DETECTED",
                "eye_closure_monitor",
                {
                    "behavior": "FATIGUE ALERT",
                    "fatigue_alert": True,
                    "average_ear": 0.18,
                    "eyes_closed_duration": 2.4,
                    "head_position": "FORWARD",
                },
            ),
            (
                "YAWNING",
                "YAWNING DETECTED",
                "mouth_temporal",
                {"yawn_event": True, "mar": 1.06},
            ),
            (
                "DISTRACTION",
                "DRIVER DISTRACTION DETECTED",
                "gaze_temporal",
                {
                    "gaze_distraction_confirmed": True,
                    "gaze_distraction_elapsed": 1.8,
                    "gaze_direction": "LEFT",
                },
            ),
            (
                "TALKING",
                "TALKING DETECTED",
                "talking_temporal",
                {
                    "talking_alert_active": True,
                    "talking_episode_count": 4,
                },
            ),
            (
                "PHONE_USE",
                "POSSIBLE DRIVER PHONE USE",
                "phone_validator",
                {
                    "driver_phone_alert": True,
                    "driver_confirmed": True,
                    "phone_detection_confirmed": True,
                    "confirmed_object_detections": [phone],
                },
            ),
            (
                "SMOKING",
                "POSSIBLE SMOKING DETECTED",
                "object_validator",
                {"confirmed_object_detections": [cigarette]},
            ),
            (
                "CUP_DETECTION",
                "CUP DETECTED",
                "object_validator",
                {"confirmed_object_detections": [cup]},
            ),
        )

        seen_event_ids = set()
        for alert_type, screenshot_title, source, updates in cases:
            with self.subTest(alert_type=alert_type):
                values = {**self.base_values(), **updates}
                call = self.run_alert(values)
                event = call.kwargs["event"]
                self.assertEqual(event.alert_type, alert_type)
                self.assertEqual(event.screenshot_title, screenshot_title)
                self.assertEqual(event.source, source)
                self.assertEqual(event.detection_frame_id, 7)
                self.assertEqual(event.screenshot_frame_id, 7)
                self.assertEqual(
                    call.args[2][0, 0, 0],
                    127,
                )
                self.assertTrue(event.confirmed)
                self.assertTrue(event.timestamp)
                self.assertNotIn(event.event_id, seen_event_ids)
                seen_event_ids.add(event.event_id)
                if alert_type == "YAWNING":
                    self.assertIn(("MAR", "1.060"), event.details)
                if alert_type == "PHONE_USE":
                    self.assertIn(
                        ("Detection confidence", "0.870"),
                        event.details,
                    )

    def test_simultaneous_yawn_and_fatigue_keep_separate_event_snapshots(self):
        worker = Mock()
        manager = AlertManager(worker, cooldown=60.0)
        first_frame = np.full((12, 12, 3), 10, dtype=np.uint8)
        values = {
            **self.base_values(),
            "behavior": "FATIGUE ALERT",
            "fatigue_alert": True,
            "eyes_closed_duration": 2.0,
            "yawn_event": True,
            "mar": 1.1,
        }
        manager.update(values, first_frame, now=10.0, frame_id=7)
        manager.update(
            {**values, "yawn_event": False, "behavior": "NORMAL",
             "fatigue_alert": False},
            np.zeros_like(first_frame),
            now=10.1,
            frame_id=8,
        )

        fatigue_event = worker.submit.call_args_list[0].kwargs["event"]
        yawn_event = worker.submit.call_args_list[1].kwargs["event"]
        self.assertEqual(fatigue_event.alert_type, "FATIGUE")
        self.assertEqual(yawn_event.alert_type, "YAWNING")
        self.assertIn(("MAR", "1.100"), yawn_event.details)
        self.assertEqual(worker.submit.call_args_list[1].args[2][0, 0, 0], 10)

    def test_gaze_and_behavior_distraction_are_one_notification_per_episode(self):
        worker = Mock()
        manager = AlertManager(worker, distraction_delay=0.0)
        values = {
            **self.base_values(),
            "behavior": "DISTRACTED",
            "gaze_distraction_confirmed": True,
            "gaze_distraction_elapsed": 2.0,
        }
        manager.update(values, "frame", now=1.0)
        manager.update(values, "frame", now=2.0)
        self.assertEqual(worker.submit.call_count, 1)
        self.assertEqual(
            worker.submit.call_args.kwargs["event"].alert_type,
            "DISTRACTION",
        )

    def test_telegram_caption_and_forwarded_screenshot_share_event_context(self):
        values = {
            **self.base_values(),
            "behavior": "FATIGUE ALERT",
            "fatigue_alert": True,
            "average_ear": 0.19,
            "eyes_closed_duration": 2.5,
            "head_position": "FORWARD",
            "phone_number": "555-0100",
        }
        telegram_worker = Mock()
        AlertManager(telegram_worker).update(values, "frame", now=1.0)
        reason, payload, frame = telegram_worker.submit.call_args.args
        event = telegram_worker.submit.call_args.kwargs["event"]

        source = SimpleNamespace(
            Config=SimpleNamespace(BOT_TOKEN="", CHAT_ID=""),
            send_telegram_alert=Mock(return_value=True),
        )
        with TemporaryDirectory() as temporary, patch.dict(
            "os.environ",
            {
                "DRIVER_MONITOR_BOT_TOKEN": "test-token",
                "DRIVER_MONITOR_CHAT_ID": "test-chat",
            },
        ):
            worker = TelegramWorker(
                threading.Event(),
                Path(temporary) / "unused-profiles.pkl",
                60.0,
                telegram_source=source,
            )
            worker.send_alert(reason, payload, frame, event=event)

        driver, caption, screenshot = (
            source.send_telegram_alert.call_args.args
        )
        kwargs = source.send_telegram_alert.call_args.kwargs
        self.assertEqual(driver, {"full_name": "Test Driver"})
        self.assertIn("DRIVER FATIGUE ALERT", caption)
        self.assertIn("Eye closure duration: 2.500 s", caption)
        self.assertNotIn("555-0100", caption)
        self.assertEqual(screenshot, frame)
        self.assertEqual(kwargs["alert_type"], "FATIGUE")
        self.assertEqual(kwargs["screenshot_title"], "FATIGUE DETECTED")
        self.assertEqual(kwargs["screenshot_frame_id"], event.screenshot_frame_id)

    def test_screenshot_renderer_uses_event_specific_colors_and_preserves_input(self):
        if telegram_source is None:
            self.fail("Active Telegram source did not load")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = {
            alert_type: telegram_source.render_alert_screenshot(
                frame,
                alert_type,
                title,
                "Validated event",
                (("Metric", "1.0"),),
                "2026-11-24T12:00:00+05:30",
                42,
            )
            for alert_type, title in (
                ("FATIGUE", "FATIGUE DETECTED"),
                ("YAWNING", "YAWNING DETECTED"),
                ("DISTRACTION", "DRIVER DISTRACTION DETECTED"),
                ("TALKING", "TALKING DETECTED"),
                ("PHONE_USE", "POSSIBLE DRIVER PHONE USE"),
                ("SMOKING", "POSSIBLE SMOKING DETECTED"),
                ("CUP_DETECTION", "CUP DETECTED"),
            )
        }
        self.assertEqual(frame.max(), 0)
        self.assertTrue(all(result.shape == frame.shape for result in results.values()))
        self.assertEqual(
            len({tuple(result[5, 5]) for result in results.values()}),
            len(results),
        )

    def test_telegram_sender_posts_event_title_and_annotated_photo(self):
        if telegram_source is None:
            self.fail("Active Telegram source did not load")
        previous_token = telegram_source.Config.BOT_TOKEN
        previous_chat = telegram_source.Config.CHAT_ID
        telegram_source.Config.BOT_TOKEN = "test-token"
        telegram_source.Config.CHAT_ID = "test-chat"
        response = SimpleNamespace(status_code=200, text="")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            with patch.object(
                telegram_source.requests,
                "post",
                side_effect=(response, response),
            ) as post:
                sent = telegram_source.send_telegram_alert(
                    {"full_name": "Test Driver", "phone": "private"},
                    "YAWNING DETECTED\n\nEvent: Yawning\nTime: 2026-11-24",
                    frame,
                    alert_type="YAWNING",
                    screenshot_title="YAWNING DETECTED",
                    screenshot_reason="Temporal yawn event confirmed.",
                    event_id="event-1",
                    timestamp="2026-11-24T12:00:00+05:30",
                    screenshot_frame_id=42,
                )
        finally:
            telegram_source.Config.BOT_TOKEN = previous_token
            telegram_source.Config.CHAT_ID = previous_chat

        self.assertTrue(sent)
        text_request = post.call_args_list[0].kwargs
        self.assertIn("YAWNING DETECTED", text_request["params"]["text"])
        photo_request = post.call_args_list[1].kwargs
        file_data = photo_request["files"]["photo"][1]
        screenshot = cv2.imdecode(
            np.frombuffer(file_data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(screenshot.shape, frame.shape)
        expected = telegram_source.TELEGRAM_ALERT_SCREENSHOT_COLORS["YAWNING"]
        self.assertTrue(
            np.allclose(screenshot[5, 5], np.asarray(expected) * 0.78, atol=8)
        )
        self.assertEqual(frame.max(), 0)


if __name__ == "__main__":
    unittest.main()
