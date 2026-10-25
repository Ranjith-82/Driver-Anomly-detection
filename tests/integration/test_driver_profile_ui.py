import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from driver_monitor.profiles.controller import DriverProfileController
from driver_monitor.profiles.model import DriverProfile
from driver_monitor.profiles.repository import DriverProfileRepository
from driver_monitor.ui.driver_profiles import DriverProfileApp


class _Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DriverProfileUiWorkflowTests(unittest.TestCase):
    def test_profile_workflow_controller_uses_temporary_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = DriverProfileController(
                DriverProfileRepository(Path(temporary) / "profiles.pkl")
            )
            controller.save(
                DriverProfile("T-01", full_name="Test Driver", email="driver@example.com"),
                is_new=True,
            )
            self.assertEqual(controller.search("test")[0].user_id, "T-01")
            controller.save(
                DriverProfile("T-01", full_name="Updated Driver", email="driver@example.com"),
                is_new=False,
            )
            self.assertEqual(controller.search("updated")[0].full_name, "Updated Driver")
            with self.assertRaises(ValueError):
                controller.save(
                    DriverProfile("T-01", email="bad-email"),
                    is_new=False,
                )
            controller.delete("T-01", confirmed=True)
            self.assertEqual(controller.search("T-01"), [])

    def test_photo_upload_copies_into_managed_storage_and_reports_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpg"
            source.write_bytes(b"test image")
            photo_dir = root / "managed-photos"
            app = SimpleNamespace(
                repository=DriverProfileRepository(
                    root / "profiles.pkl", managed_photo_dir=photo_dir
                ),
                user_id=_Value("D/01"),
                photo_status=_Value(),
                _show_photo=lambda path: setattr(app, "previewed_path", path),
            )
            with patch(
                "driver_monitor.ui.driver_profiles.filedialog.askopenfilename",
                return_value=str(source),
            ):
                DriverProfileApp._upload_photo(app)

            self.assertTrue(app.photo_source.is_file())
            self.assertEqual(app.photo_source.parent, photo_dir)
            self.assertTrue(app.photo_source.name.startswith("D_01-"))
            self.assertTrue(app.photo_status.get().startswith("Selected:"))
            self.assertEqual(app.previewed_path, app.photo_source)

    def test_photo_capture_saves_image_and_reports_status(self):
        class Capture:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def read(self):
                return True, SimpleNamespace(copy=lambda: b"captured frame")

            def release(self):
                self.released = True

        capture = Capture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo_dir = root / "managed-photos"
            cv2 = SimpleNamespace(
                VideoCapture=lambda _index: capture,
                imshow=lambda *_args: None,
                waitKey=lambda _delay: 32,
                destroyWindow=lambda _name: None,
                imwrite=lambda path, frame: Path(path).write_bytes(frame) > 0,
            )
            app = SimpleNamespace(
                repository=DriverProfileRepository(
                    root / "profiles.pkl", managed_photo_dir=photo_dir
                ),
                user_id=_Value("D-02"),
                photo_status=_Value(),
                _show_photo=lambda path: setattr(app, "previewed_path", path),
            )
            with patch.dict(sys.modules, {"cv2": cv2}):
                DriverProfileApp._capture_photo(app)

            self.assertTrue(capture.released)
            self.assertTrue(app.photo_source.is_file())
            self.assertTrue(app.photo_source.name.startswith("D-02-"))
            self.assertTrue(app.photo_status.get().startswith("Captured:"))
            self.assertEqual(app.previewed_path, app.photo_source)

    def test_delete_requires_confirmation_before_repository_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = DriverProfileRepository(Path(temporary) / "profiles.pkl")
            repository.create(DriverProfile("T-02", full_name="Test Driver"))
            app = SimpleNamespace(
                current_user_id="T-02",
                controller=DriverProfileController(repository),
                _new_profile=lambda: None,
                _refresh_profiles=lambda: None,
            )
            with patch(
                "driver_monitor.ui.driver_profiles.messagebox.askyesno",
                return_value=False,
            ):
                DriverProfileApp._delete_profile(app)
            self.assertIsNotNone(repository.get("T-02"))

            with patch(
                "driver_monitor.ui.driver_profiles.messagebox.askyesno",
                return_value=True,
            ):
                DriverProfileApp._delete_profile(app)
            self.assertIsNone(repository.get("T-02"))


if __name__ == "__main__":
    unittest.main()
