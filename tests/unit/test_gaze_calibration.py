import sys
from types import ModuleType
import unittest

import cv2
import numpy as np

from driver_monitor.legacy import load_gaze_module
from driver_monitor.monitoring.legacy_runtime import (
    GAZE_CALIBRATION_PANEL_WIDTH,
    draw_gaze_calibration_screen,
    handle_gaze_calibration_click,
)


class GazeCalibrationTests(unittest.TestCase):
    def setUp(self):
        needs_mediapipe_stub = "mediapipe" not in sys.modules
        if needs_mediapipe_stub:
            sys.modules["mediapipe"] = ModuleType("mediapipe")
        try:
            self.module = load_gaze_module()
        finally:
            if needs_mediapipe_stub:
                del sys.modules["mediapipe"]
        self.session = self.module.CalibrationSession()

    def _capture(self, x, y, start):
        self.session.start_capture(start)
        for index in range(self.module.CALIBRATION_MIN_SAMPLES):
            sample_time = start + (
                self.module.CALIBRATION_CAPTURE_SECONDS
                * (index + 1) / self.module.CALIBRATION_MIN_SAMPLES
            )
            self.session.update(
                sample_time, self.module.GazePosition(x=x, y=y)
            )

    def test_retake_replaces_selected_sample_and_restores_pending_step(self):
        self._capture(0.5, 0.5, 0.0)
        self.session.start_capture(2.0)
        self._capture(0.4, 0.5, 3.0)
        self.session.start_capture(5.0)
        self.assertEqual(self.session.current_direction, "RIGHT")

        self.assertTrue(self.session.retake("FORWARD"))
        self.assertEqual(self.session.current_direction, "FORWARD")
        self._capture(0.3, 0.6, 6.0)
        self.session.start_capture(8.0)

        self.assertEqual(self.session.current_direction, "RIGHT")
        self.assertFalse(self.session.captured)
        self.assertAlmostEqual(self.session.observations["FORWARD"].x, 0.3)
        self.assertAlmostEqual(self.session.observations["LEFT"].x, 0.4)

    def test_retake_restores_a_sampled_pending_step(self):
        self._capture(0.5, 0.5, 0.0)
        self.session.start_capture(2.0)
        self._capture(0.4, 0.5, 3.0)

        self.assertTrue(self.session.retake("FORWARD"))
        self._capture(0.3, 0.6, 5.0)
        self.session.start_capture(7.0)

        self.assertEqual(self.session.current_direction, "LEFT")
        self.assertTrue(self.session.captured)
        self.session.start_capture(8.0)
        self.assertEqual(self.session.current_direction, "RIGHT")

    def test_retake_rejects_uncaptured_or_currently_capturing_samples(self):
        self.assertFalse(self.session.retake("FORWARD"))
        self.session.start_capture(0.0)
        self.assertFalse(self.session.retake("FORWARD"))

    def test_calibration_screen_keeps_text_panel_and_video_side_by_side(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        position = self.module.GazePosition(x=0.5, y=0.5)

        screen = draw_gaze_calibration_screen(
            frame, self.session, position, now=0.0
        )

        self.assertEqual(screen.shape, (480, GAZE_CALIBRATION_PANEL_WIDTH + 640, 3))
        np.testing.assert_array_equal(screen[0, 0], (18, 25, 30))
        np.testing.assert_array_equal(
            screen[200, GAZE_CALIBRATION_PANEL_WIDTH + 20], (255, 255, 255)
        )

    def test_clicking_captured_direction_starts_its_retake(self):
        self._capture(0.5, 0.5, 0.0)

        handle_gaze_calibration_click(
            cv2.EVENT_LBUTTONDOWN, 40, 264 + 17, 0, self.session
        )

        self.assertFalse(self.session.captured)
        self.assertEqual(self.session.current_direction, "FORWARD")


if __name__ == "__main__":
    unittest.main()
