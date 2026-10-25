import pickle
import tempfile
import unittest
from pathlib import Path

from driver_monitor.profiles.model import DriverProfile
from driver_monitor.profiles.repository import DriverProfileRepository


class DriverProfileRepositoryTests(unittest.TestCase):
    def test_create_search_update_and_delete_with_temporary_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = DriverProfileRepository(root / "drivers_database.pkl")
            profile = DriverProfile("001", full_name="Alex Driver", phone="00123")
            repository.create(profile)
            self.assertEqual(repository.search("alex")[0].user_id, "001")
            with self.assertRaises(ValueError):
                repository.create(profile)
            repository.update(DriverProfile("001", full_name="Alex Updated"))
            self.assertEqual(repository.get("001").full_name, "Alex Updated")
            with self.assertRaises(PermissionError):
                repository.delete("001")
            repository.delete("001", confirm=True)
            self.assertIsNone(repository.get("001"))

    def test_loads_legacy_pickle_and_preserves_photo_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "drivers_database.pkl"
            legacy = {
                "A1": {
                    "user_id": "A1",
                    "full_name": "Legacy Driver",
                    "photo": "driver_photos/A1.jpg",
                    "license_number": "0001",
                    "custom_field": "preserve",
                }
            }
            with database.open("wb") as file:
                pickle.dump(legacy, file)
            repository = DriverProfileRepository(database)
            profile = repository.get("A1")
            self.assertEqual(profile.photo_reference, "driver_photos/A1.jpg")
            repository.update(DriverProfile(
                "A1",
                full_name="Updated",
                license_number="0001",
                photo_reference="driver_photos/A1.jpg",
            ))
            with database.open("rb") as file:
                saved = pickle.load(file)
            self.assertEqual(saved["A1"]["photo"], "driver_photos/A1.jpg")
            self.assertEqual(saved["A1"]["license_number"], "0001")
            self.assertEqual(saved["A1"]["custom_field"], "preserve")

    def test_deletes_only_photo_under_managed_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            managed = root / "driver_photos"
            managed.mkdir()
            photo = managed / "A1.jpg"
            photo.touch()
            repository = DriverProfileRepository(
                root / "drivers_database.pkl", managed_photo_dir=managed
            )
            repository.create(DriverProfile("A1", photo_reference=str(photo)))
            repository.delete("A1", confirm=True, remove_managed_photo=True)
            self.assertFalse(photo.exists())


if __name__ == "__main__":
    unittest.main()
