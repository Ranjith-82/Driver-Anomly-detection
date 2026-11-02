"""Compatibility launcher for the packaged driver profile manager."""

import sys
from pathlib import Path

legacy_directory = Path(__file__).resolve().parent
if str(legacy_directory) not in sys.path:
    sys.path.insert(0, str(legacy_directory))

from driver_app_legacy import DriverApp


def main() -> None:
    repository_root = legacy_directory.parents[1]
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from driver_monitor.ui.driver_profiles import run_profile_manager

    run_profile_manager()


if __name__ == "__main__":
    main()
