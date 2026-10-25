"""Compatibility launcher for the packaged buffered driver monitor."""

import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[5]
source_root = repository_root / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from driver_monitor.monitoring.legacy_runtime import main


if __name__ == "__main__":
    main()
