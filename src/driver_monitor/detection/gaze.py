from driver_monitor.legacy import load_gaze_module


def create_calibration_session():
    return load_gaze_module().CalibrationSession()


def create_gaze_smoother():
    return load_gaze_module().GazeSmoother()


def create_distraction_timer(duration):
    return load_gaze_module().DistractionTimer(duration)


def extract_eye_landmarks(landmarks, width, height):
    return load_gaze_module().extract_eye_landmarks(landmarks, width, height)


def calculate_gaze_position(eye_landmarks):
    return load_gaze_module().calculate_gaze_position(eye_landmarks)


def classify_gaze(position, thresholds):
    return load_gaze_module().classify_gaze(position, thresholds)


def calibration_directions():
    return load_gaze_module().GAZE_DIRECTIONS


def calibration_capture_seconds():
    return load_gaze_module().CALIBRATION_CAPTURE_SECONDS


def calibrate_gaze_thresholds(observations):
    return load_gaze_module().calibrate_gaze_thresholds(observations)
