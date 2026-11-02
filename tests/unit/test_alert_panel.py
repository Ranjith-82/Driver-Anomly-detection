import io
import unittest
import wave
from unittest.mock import Mock, patch

import numpy as np

from driver_monitor.monitoring.alerts import AlertManager
from driver_monitor.monitoring.sound import (
    AlertAudioManager,
    AlertAudioWorker,
    DISTRACTION_BEEP_DURATION,
    DISTRACTION_BEEP_FREQUENCY,
    DISTRACTION_MESSAGE,
    FATIGUE_BEEP_COOLDOWN,
    FATIGUE_BEEP_DURATION,
    FATIGUE_BEEP_FREQUENCY,
    FATIGUE_MESSAGE,
    TALKING_ALERT_COOLDOWN,
    YAWN_BEEP_DURATION,
    YAWN_BEEP_FREQUENCY,
    _tone_wave,
)
from driver_monitor.ui.alert_panel import render_alert_panel


class AlertPanelTests(unittest.TestCase):
    def test_alert_panel_renders_expected_dimensions(self):
        panel = render_alert_panel(
            260,
            618,
            "FATIGUE ALERT",
            face_present=True,
        )
        self.assertEqual(panel.shape, (618, 260, 3))
        self.assertEqual(panel.dtype, np.uint8)
        self.assertEqual(tuple(panel[1, 1]), (40, 95, 235))


class AlertAudioSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.worker = Mock()
        self.manager = AlertAudioManager(self.worker)

    @staticmethod
    def values(behavior="NORMAL", **updates):
        values = {
            "behavior": behavior,
            "face_present": True,
            "fatigue_alert": behavior == "FATIGUE ALERT",
            "yawn_event": False,
        }
        values.update(updates)
        return values

    def test_fatigue_speaks_with_45_second_cooldown(self):
        self.manager.update(self.values("FATIGUE ALERT"), 10.0)
        self.worker.enqueue_voice.assert_called_once_with(
            "fatigue",
            FATIGUE_MESSAGE,
            cancel_when_inactive=True,
        )
        self.manager.update(self.values("FATIGUE ALERT"), 54.9)
        self.assertEqual(self.worker.enqueue_voice.call_count, 1)
        self.manager.update(self.values("FATIGUE ALERT"), 55.0)
        self.assertEqual(self.worker.enqueue_voice.call_count, 2)

    def test_eye_closure_threshold_queues_immediate_beep_without_other_events(self):
        values = self.values(
            "NORMAL",
            fatigue_alert=False,
            eyes_closed_duration=1.5,
            yawn_event=False,
            talking=False,
            distracted=False,
            emotion_label="unknown",
        )
        self.manager.update(values, 10.0)

        self.worker.enqueue_beeps.assert_called_once_with(
            "fatigue",
            FATIGUE_BEEP_FREQUENCY,
            FATIGUE_BEEP_DURATION,
            count=1,
            gap_ms=120,
            cancel_when_inactive=False,
        )
        self.worker.enqueue_voice.assert_called_once_with(
            "fatigue",
            FATIGUE_MESSAGE,
            cancel_when_inactive=True,
        )

    def test_open_or_below_threshold_eyes_do_not_queue_fatigue_beep(self):
        for now, duration in ((1.0, 0.0), (2.0, 1.49)):
            self.manager.update(
                self.values(
                    "NORMAL",
                    fatigue_alert=False,
                    eyes_closed_duration=duration,
                ),
                now,
            )
        self.worker.enqueue_beeps.assert_not_called()
        self.worker.enqueue_voice.assert_not_called()

    def test_fatigue_beep_is_cooldown_limited_and_repeats(self):
        values = self.values(
            "NORMAL", fatigue_alert=False, eyes_closed_duration=1.5
        )
        self.manager.update(values, 10.0)
        self.manager.update(values, 10.1)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 1)

        self.manager.update(values, 10.0 + FATIGUE_BEEP_COOLDOWN - 0.01)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 1)
        self.manager.update(values, 10.0 + FATIGUE_BEEP_COOLDOWN)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 2)

    def test_reclosed_eyes_can_beep_again_after_cooldown(self):
        closed = self.values(
            "NORMAL", fatigue_alert=False, eyes_closed_duration=1.5
        )
        open_eyes = self.values(
            "NORMAL", fatigue_alert=False, eyes_closed_duration=0.0
        )
        self.manager.update(closed, 10.0)
        self.manager.update(open_eyes, 11.0)
        self.manager.update(
            closed, 10.0 + FATIGUE_BEEP_COOLDOWN
        )
        self.assertEqual(self.worker.enqueue_beeps.call_count, 2)

    def test_fatigue_beep_uses_configured_eye_closure_duration(self):
        manager = AlertAudioManager(self.worker, fatigue_seconds=2.0)
        manager.update(
            self.values(
                "NORMAL",
                fatigue_alert=False,
                eyes_closed_duration=1.99,
            ),
            1.0,
        )
        self.worker.enqueue_beeps.assert_not_called()
        manager.update(
            self.values(
                "NORMAL",
                fatigue_alert=False,
                eyes_closed_duration=2.0,
            ),
            2.0,
        )
        self.worker.enqueue_beeps.assert_called_once_with(
            "fatigue",
            FATIGUE_BEEP_FREQUENCY,
            FATIGUE_BEEP_DURATION,
            count=1,
            gap_ms=120,
            cancel_when_inactive=False,
        )

    def test_existing_fatigue_telegram_notification_still_requires_three_events(self):
        telegram = Mock()
        alerts = AlertManager(telegram)
        values = {
            "behavior": "NORMAL",
            "face_present": True,
            "fatigue_alert": True,
            "eyes_closed_duration": 1.5,
            "yawn_event": False,
            "object_detections": [],
        }
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for timestamp in (1.0, 2.0, 3.0):
            alerts.update(values, frame, timestamp)
            alerts.update({**values, "fatigue_alert": False}, frame, timestamp + 0.1)

        telegram.submit.assert_called_once()
        self.assertEqual(telegram.submit.call_args.args[0], "FATIGUE ALERT")

    def test_fatigue_takes_priority_over_simultaneous_distraction_state(self):
        self.manager.update(
            self.values(
                "DISTRACTED",
                fatigue_alert=True,
                gaze_distraction_confirmed=True,
            ),
            1.0,
        )
        self.worker.enqueue_voice.assert_called_once_with(
            "fatigue",
            FATIGUE_MESSAGE,
            cancel_when_inactive=True,
        )
        self.worker.enqueue_beeps.assert_called_once_with(
            "fatigue",
            FATIGUE_BEEP_FREQUENCY,
            FATIGUE_BEEP_DURATION,
            count=1,
            gap_ms=120,
            cancel_when_inactive=False,
        )

    def test_distraction_waits_for_delay_and_repeats_on_configured_intervals(self):
        for now in (0.0, 1.49):
            self.manager.update(self.values("DISTRACTED"), now)
        self.worker.enqueue_beeps.assert_not_called()
        self.worker.enqueue_voice.assert_not_called()

        self.manager.update(self.values("DISTRACTED"), 1.5)
        self.worker.enqueue_beeps.assert_called_once_with(
            "distraction",
            DISTRACTION_BEEP_FREQUENCY,
            DISTRACTION_BEEP_DURATION,
            count=1,
            gap_ms=120,
            cancel_when_inactive=True,
        )
        self.worker.enqueue_voice.assert_called_once_with(
            "distraction",
            DISTRACTION_MESSAGE,
            cancel_when_inactive=True,
        )

        self.manager.update(self.values("DISTRACTED"), 5.49)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 1)
        self.manager.update(self.values("DISTRACTED"), 5.5)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 2)
        self.assertEqual(self.worker.enqueue_voice.call_count, 2)

        self.manager.update(self.values("NORMAL"), 5.6)
        self.manager.update(self.values("DISTRACTED"), 8.0)
        self.manager.update(self.values("DISTRACTED"), 9.5)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 3)
        self.assertEqual(self.worker.enqueue_voice.call_count, 3)

    def test_distraction_is_not_triggered_without_face(self):
        values = self.values("DISTRACTED", face_present=False)
        self.manager.update(values, 0)
        self.manager.update(values, 10)
        self.worker.enqueue_beeps.assert_not_called()
        self.worker.enqueue_voice.assert_not_called()

    def test_confirmed_gaze_distraction_uses_distraction_audio_schedule(self):
        values = self.values(
            "NORMAL",
            gaze_distraction_confirmed=True,
        )
        self.manager.update(values, 0)
        self.manager.update(values, 1.5)
        self.worker.enqueue_beeps.assert_called_once()
        self.worker.enqueue_voice.assert_called_once_with(
            "distraction",
            DISTRACTION_MESSAGE,
            cancel_when_inactive=True,
        )

    def test_yawn_event_schedules_two_beeps_with_ten_second_cooldown(self):
        values = self.values("NORMAL", yawn_event=True)
        self.manager.update(values, 1.0)
        self.manager.update(values, 5.0)
        self.worker.enqueue_beeps.assert_called_once_with(
            "yawn",
            YAWN_BEEP_FREQUENCY,
            YAWN_BEEP_DURATION,
            count=2,
            gap_ms=120,
            cancel_when_inactive=False,
        )
        self.manager.update(values, 10.99)
        self.worker.enqueue_beeps.assert_called_once()
        self.manager.update(values, 11.0)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 2)
        self.worker.enqueue_voice.assert_not_called()

    def test_talking_speaks_after_four_episodes_and_respects_two_minutes(self):
        timestamp = 0.0
        for _ in range(3):
            self.manager.update(self.values("TALKING"), timestamp)
            timestamp += 0.1
            self.manager.update(self.values("NORMAL"), timestamp)
            timestamp += 0.1
        self.worker.enqueue_voice.assert_not_called()

        self.manager.update(self.values("TALKING"), timestamp)
        self.worker.enqueue_voice.assert_called_once()
        self.assertEqual(
            self.worker.enqueue_voice.call_args.args[0],
            "talking",
        )
        self.assertFalse(
            self.worker.enqueue_voice.call_args.kwargs[
                "cancel_when_inactive"
            ]
        )

        self.manager.update(self.values("NORMAL"), timestamp + 1)
        self.manager.update(self.values("TALKING"), timestamp + 2)
        self.assertEqual(self.worker.enqueue_voice.call_count, 1)

        self.manager.update(self.values("NORMAL"), timestamp + 3)
        self.manager.update(
            self.values("TALKING"),
            timestamp + TALKING_ALERT_COOLDOWN + 1,
        )
        self.assertEqual(self.worker.enqueue_voice.call_count, 2)

    def test_object_alert_plays_once_until_detection_clears(self):
        values = self.values(
            "NORMAL",
            object_detections=[{"class_id": 0}],
            driver_confirmed=True,
            phone_detection_confirmed=True,
            driver_phone_alert=True,
        )
        self.manager.update(values, 0)
        self.manager.update(values, 1)
        self.worker.enqueue_beeps.assert_called_once_with(
            "object_phone",
            DISTRACTION_BEEP_FREQUENCY,
            DISTRACTION_BEEP_DURATION,
            count=1,
            gap_ms=120,
            cancel_when_inactive=False,
        )
        self.worker.enqueue_voice.assert_called_once_with(
            "object_phone",
            "Possible phone use near the identified driver. "
            "Please keep the phone away while driving.",
            cancel_when_inactive=False,
        )
        self.manager.update(self.values("NORMAL"), 2)
        self.manager.update(values, 3)
        self.assertEqual(self.worker.enqueue_beeps.call_count, 2)
        self.assertEqual(self.worker.enqueue_voice.call_count, 2)

    def test_unconfirmed_or_unidentified_phone_does_not_trigger_driver_alert(self):
        self.manager.update(
            self.values(
                "NORMAL",
                object_detections=[{"class_id": 0}],
            ),
            0,
        )
        self.worker.enqueue_beeps.assert_not_called()
        self.worker.enqueue_voice.assert_not_called()

        self.manager.update(
            self.values(
                "NORMAL",
                object_detections=[{"class_id": 0}],
                phone_detection_confirmed=True,
                driver_phone_alert=True,
                driver_confirmed=False,
            ),
            1,
        )
        self.worker.enqueue_beeps.assert_not_called()
        self.worker.enqueue_voice.assert_not_called()

    def test_muted_manager_does_not_schedule_voice_or_beeps(self):
        manager = AlertAudioManager(self.worker, enabled=False)
        with patch("builtins.print"):
            manager.update(self.values("FATIGUE ALERT"), 1.0)
            manager.update(self.values("DISTRACTED"), 10.0)
            manager.update(self.values("DISTRACTED"), 11.5)
        self.worker.enqueue_voice.assert_not_called()
        self.worker.enqueue_beeps.assert_not_called()


class AudioPlaybackTests(unittest.TestCase):
    def test_generated_beep_is_a_valid_pcm_wave(self):
        audio = _tone_wave(900, 180)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 44100)
            self.assertAlmostEqual(wav.getnframes() / 44100, 0.18, places=2)

    def test_yawn_waveform_contains_two_tones_and_gap(self):
        audio = _tone_wave(650, 120, count=2, gap_ms=120)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertAlmostEqual(wav.getnframes() / 44100, 0.36, places=2)

    def test_worker_mute_is_explicit_and_does_not_start_playback(self):
        worker = AlertAudioWorker(muted=True)
        with patch("builtins.print") as output:
            worker.run()
        output.assert_called_once_with("[AUDIO] Audio disabled by --mute.")

    def test_audio_events_are_deduplicated_until_processed(self):
        worker = AlertAudioWorker()
        with patch("builtins.print"):
            self.assertTrue(worker.enqueue_voice("fatigue", "Warning"))
            self.assertFalse(worker.enqueue_voice("fatigue", "Warning"))
        self.assertEqual(worker._queue.qsize(), 1)
        worker.close()


if __name__ == "__main__":
    unittest.main()
