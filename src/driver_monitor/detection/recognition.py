from pathlib import Path

from driver_monitor.legacy import load_recognition_module
from driver_monitor.profiles.repository import DriverProfileRepository


def create_live_recognizer(
    profiles: DriverProfileRepository | Path,
):
    database_path = (
        profiles.database_path
        if isinstance(profiles, DriverProfileRepository)
        else Path(profiles)
    )
    module = load_recognition_module()
    return module.LiveDriverRecognition(str(database_path))


def recognize_frame(
    frame,
    recognizer,
    *,
    tolerance: float,
    resize_scale: float,
):
    module = load_recognition_module()
    small_frame = module.cv2.resize(
        frame, (0, 0), fx=resize_scale, fy=resize_scale
    )
    rgb_small_frame = module.cv2.cvtColor(
        small_frame, module.cv2.COLOR_BGR2RGB
    )
    locations = module.face_recognition.face_locations(
        rgb_small_frame, model="hog"
    )
    encodings = module.face_recognition.face_encodings(
        rgb_small_frame, locations
    )
    results = []
    for encoding, location in zip(encodings, locations):
        distances = module.face_recognition.face_distance(
            recognizer.known_encodings, encoding
        )
        if len(distances) == 0:
            results.append((None, 0.0, location))
            continue
        best_index = int(module.np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance < tolerance:
            results.append(
                (
                    recognizer.known_driver_ids[best_index],
                    1.0 - best_distance,
                    location,
                )
            )
        else:
            results.append((None, 0.0, location))
    return results
