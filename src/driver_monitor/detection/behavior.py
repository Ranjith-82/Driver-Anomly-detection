from driver_monitor.legacy import load_behavior_module


def create_landmark_detector(model_path=None):
    return load_behavior_module().FaceLandmarkDetector(model_path)


def create_mouth_analyzer(**options):
    return load_behavior_module().MouthAnalyzer(**options)


def create_eye_analyzer(**options):
    return load_behavior_module().EyeAnalyzer(**options)


def create_behavior_state_machine(**options):
    return load_behavior_module().BehaviorStateMachine(**options)


def create_head_pose_estimator():
    return load_behavior_module().HeadPoseEstimator()


def corrected_head_pose(landmarks, width, height):
    return load_behavior_module().corrected_head_pose(landmarks, width, height)


def mouth_features(landmarks):
    return load_behavior_module().mouth_features(landmarks)


def extract_behavior_features(landmarks, width, height):
    return load_behavior_module().feature_vector(landmarks, width, height)
