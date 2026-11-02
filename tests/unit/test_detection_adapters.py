import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from driver_monitor.detection import behavior, gaze


class DetectionAdapterTests(unittest.TestCase):
    def test_behavior_adapter_delegates_constructors_and_features(self):
        legacy = SimpleNamespace(
            FaceLandmarkDetector=Mock(return_value="landmarker"),
            MouthAnalyzer=Mock(return_value="mouth"),
            EyeAnalyzer=Mock(return_value="eyes"),
            BehaviorStateMachine=Mock(return_value="state"),
            HeadPoseEstimator=Mock(return_value="pose"),
            corrected_head_pose=Mock(return_value=(None, "rotation")),
            mouth_features=Mock(return_value=0.5),
            feature_vector=Mock(return_value=[1.0, 2.0]),
        )
        with patch(
            "driver_monitor.detection.behavior.load_behavior_module",
            return_value=legacy,
        ):
            self.assertEqual(behavior.create_landmark_detector("face.task"), "landmarker")
            self.assertEqual(behavior.create_mouth_analyzer(mar_threshold=0.6), "mouth")
            self.assertEqual(behavior.create_eye_analyzer(close_threshold=0.2), "eyes")
            self.assertEqual(
                behavior.create_behavior_state_machine(yaw_threshold=25),
                "state",
            )
            self.assertEqual(behavior.create_head_pose_estimator(), "pose")
            self.assertEqual(behavior.corrected_head_pose([], 640, 480), (None, "rotation"))
            self.assertEqual(behavior.mouth_features([]), 0.5)
            self.assertEqual(
                behavior.extract_behavior_features([], 640, 480),
                [1.0, 2.0],
            )
        legacy.FaceLandmarkDetector.assert_called_once_with("face.task")

    def test_gaze_adapter_delegates_calibration_and_runtime_calls(self):
        legacy = SimpleNamespace(
            CalibrationSession=Mock(return_value="calibration"),
            GazeSmoother=Mock(return_value="smoother"),
            DistractionTimer=Mock(return_value="timer"),
            extract_eye_landmarks=Mock(return_value=["eyes"]),
            calculate_gaze_position=Mock(return_value="position"),
            classify_gaze=Mock(return_value="FORWARD"),
            GAZE_DIRECTIONS=("FORWARD", "LEFT"),
            CALIBRATION_CAPTURE_SECONDS=1.5,
            calibrate_gaze_thresholds=Mock(return_value="thresholds"),
        )
        with patch(
            "driver_monitor.detection.gaze.load_gaze_module",
            return_value=legacy,
        ):
            self.assertEqual(gaze.create_calibration_session(), "calibration")
            self.assertEqual(gaze.create_gaze_smoother(), "smoother")
            self.assertEqual(gaze.create_distraction_timer(0.6), "timer")
            self.assertEqual(gaze.extract_eye_landmarks([], 640, 480), ["eyes"])
            self.assertEqual(gaze.calculate_gaze_position(["eyes"]), "position")
            self.assertEqual(gaze.classify_gaze("position", "thresholds"), "FORWARD")
            self.assertEqual(gaze.calibration_directions(), ("FORWARD", "LEFT"))
            self.assertEqual(gaze.calibration_capture_seconds(), 1.5)
            self.assertEqual(
                gaze.calibrate_gaze_thresholds({"FORWARD": "sample"}),
                "thresholds",
            )
        legacy.DistractionTimer.assert_called_once_with(0.6)


if __name__ == "__main__":
    unittest.main()
