import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from driver_monitor import app
from driver_monitor.config import ProjectPaths
from driver_monitor.ui.driver_profiles import run_profile_manager


class LegacyEntryPointTests(unittest.TestCase):
    def test_profile_manager_and_live_monitor_entrypoints_are_resolvable(self):
        paths = ProjectPaths.discover()
        self.assertTrue(paths.legacy_monitor_dir.joinpath(
            "buffered_driver_monitor.py").is_file())
        self.assertTrue(callable(run_profile_manager))

    def test_packaged_runtime_import_does_not_require_recognition_extra(self):
        runtime = app._load_monitor_runtime()
        self.assertTrue(callable(runtime.main))

    def test_package_entrypoint_forwards_legacy_monitor_options(self):
        forwarded_argv = []
        legacy = SimpleNamespace(main=lambda: forwarded_argv.extend(app.sys.argv))
        with patch.object(app, "_load_monitor_runtime", return_value=legacy), patch.object(
            app.sys, "argv", ["driver_monitor"]
        ):
            app.main(["--width", "960", "--benchmark"])
        expected_script = str(
            ProjectPaths.discover().legacy_monitor_dir
            / "buffered_driver_monitor.py"
        )
        self.assertEqual(
            forwarded_argv,
            [expected_script, "--width", "960", "--benchmark"],
        )

    def test_profile_entrypoint_uses_profile_manager(self):
        with patch(
            "driver_monitor.ui.driver_profiles.run_profile_manager"
        ) as run_manager:
            app.main(["--profiles"])
        run_manager.assert_called_once_with()

    def test_phase_two_profile_launcher_delegates_to_package_ui(self):
        launcher_path = (
            ProjectPaths.discover().legacy_profiles_dir / "driver_app.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_legacy_driver_app", launcher_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        self.assertTrue(callable(launcher.DriverApp))
        with patch(
            "driver_monitor.ui.driver_profiles.run_profile_manager"
        ) as run_manager:
            launcher.main()
        run_manager.assert_called_once_with()

    def test_phase_three_legacy_launcher_delegates_to_package_runtime(self):
        launcher_path = (
            ProjectPaths.discover().legacy_monitor_dir
            / "buffered_driver_monitor.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_legacy_buffered_launcher", launcher_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        launcher = importlib.util.module_from_spec(spec)
        runtime = SimpleNamespace(main=Mock())
        with patch.dict(
            sys.modules,
            {"driver_monitor.monitoring.legacy_runtime": runtime},
        ):
            spec.loader.exec_module(launcher)
            launcher.main()
        runtime.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
