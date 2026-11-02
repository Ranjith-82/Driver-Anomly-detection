from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

from driver_monitor.config import ProjectPaths


def _load_monitor_runtime():
    paths = ProjectPaths.discover()
    if not paths.legacy_monitor_dir.is_dir():
        raise FileNotFoundError(
            f"Legacy monitoring dependencies not found: {paths.legacy_monitor_dir}"
        )
    return importlib.import_module("driver_monitor.monitoring.legacy_runtime")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="driver_monitor",
        description="Run the existing buffered driver monitor or manage driver profiles.",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Open the driver profile manager instead of the live monitor.",
    )
    args, legacy_args = parser.parse_known_args(argv)

    if args.profiles:
        from driver_monitor.ui.driver_profiles import run_profile_manager

        run_profile_manager()
        return

    runtime = _load_monitor_runtime()
    previous_argv = sys.argv
    try:
        sys.argv = [str(ProjectPaths.discover().legacy_monitor_dir / "buffered_driver_monitor.py")]
        sys.argv.extend(legacy_args)
        runtime.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
