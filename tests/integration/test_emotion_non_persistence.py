import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from driver_monitor.models import EmotionLabel
from driver_monitor.monitoring.alerts import notification_values
from driver_monitor.monitoring.session import LiveMonitoringSession
from driver_monitor.profiles.model import DriverProfile
from driver_monitor.profiles.repository import DriverProfileRepository


class EmotionNonPersistenceTests(unittest.TestCase):
    def test_session_shutdown_clears_results_and_latest_frame(self):
        session = LiveMonitoringSession()
        session.results.update("emotion", EmotionLabel.ANGER)
        session.frames.put(type("Packet", (), {"frame_id": 1})())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            session.stop()
        self.assertEqual(session.results.snapshot(), {})
        self.assertEqual(session.frames._packet, None)
        self.assertNotIn(EmotionLabel.ANGER.value, output.getvalue())

    def test_profile_storage_and_notifications_exclude_emotion_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "profiles.pkl"
            repository = DriverProfileRepository(database)
            profile = DriverProfile("test-driver", full_name="Test Driver")
            repository.create(profile)
            notification = notification_values({
                "driver_name": "Test Driver",
                "emotion_label": "anger",
                "emotion_status": "ready",
                "emotion_error": "",
            })

            self.assertNotIn("emotion_label", notification)
            self.assertNotIn("emotion_status", notification)
            self.assertNotIn("emotion_error", notification)
            self.assertEqual(repository.get("test-driver"), profile)
            self.assertEqual(list(Path(temporary).iterdir()), [database])


if __name__ == "__main__":
    unittest.main()
