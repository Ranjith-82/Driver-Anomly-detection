import hashlib
import tempfile
import unittest
from pathlib import Path

from driver_monitor.config import ProjectPaths


class ProjectPathTests(unittest.TestCase):
    def test_model_paths_use_organized_model_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ProjectPaths.discover(root)
            self.assertEqual(paths.repository_root, root.resolve())
            self.assertEqual(
                paths.behavior_model,
                root / "models" / "behavior" / "driver_behavior_model.pkl",
            )
            self.assertEqual(
                paths.behavior_labels,
                root / "models" / "behavior" / "driver_behavior_labels.pkl",
            )
            self.assertEqual(
                paths.face_landmarker,
                root / "models" / "face_landmarker" / "face_landmarker.task",
            )
            self.assertEqual(
                paths.object_model,
                root / "models" / "objects" / "best_phone_finetuned_v1.pt",
            )
            self.assertEqual(paths.driver_database, root / "data" / "local"
                             / "drivers_database.pkl")
            self.assertTrue(paths.driver_photos.is_relative_to(paths.local_data_dir))

    def test_migrated_artifacts_match_legacy_artifacts(self):
        paths = ProjectPaths.discover()
        pairs = (
            (
                paths.behavior_model,
                paths.repository_root / "Phase-3" / "Buffered-system"
                / "buffered_system" / "Buffer" / "Buffered_system"
                / "driver_behavior_model.pkl",
            ),
            (
                paths.behavior_labels,
                paths.repository_root / "Phase-3" / "Buffered-system"
                / "buffered_system" / "Buffer" / "Buffered_system"
                / "driver_behavior_labels.pkl",
            ),
            (
                paths.face_landmarker,
                paths.repository_root / "Phase-2" / "Gaze_detection"
                / "face_landmarker.task",
            ),
            (
                paths.object_model,
                paths.repository_root / "Phase-3" / "Buffered-system"
                / "buffered_system" / "Buffer" / "Buffered_system"
                / "best_phone_finetuned_v1.pt",
            ),
        )
        for migrated, legacy in pairs:
            with self.subTest(artifact=migrated.name):
                self.assertTrue(migrated.is_file())
                self.assertTrue(legacy.is_file())
                self.assertEqual(migrated.stat().st_size, legacy.stat().st_size)
                self.assertEqual(
                    hashlib.sha256(migrated.read_bytes()).digest(),
                    hashlib.sha256(legacy.read_bytes()).digest(),
                )

    def test_existing_legacy_profile_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "Phase-2" / "Face_recognition_system"
            legacy.mkdir(parents=True)
            (legacy / "drivers_database.pkl").touch()
            (legacy / "driver_photos").mkdir()
            paths = ProjectPaths.discover(root)
            self.assertEqual(paths.driver_database, legacy / "drivers_database.pkl")
            self.assertEqual(paths.driver_photos, legacy / "driver_photos")


if __name__ == "__main__":
    unittest.main()
