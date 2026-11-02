from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from driver_monitor.config import ProjectPaths


def load_legacy_module(name: str, source: Path) -> ModuleType:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy module source not found: {source}")
    directory = str(source.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module_name = f"driver_monitor_legacy_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_behavior_module() -> ModuleType:
    paths = ProjectPaths.discover()
    source = paths.repository_root / "Phase-2" / "Face_monitoring_system" / (
        "train_and_run_driver_detector.py"
    )
    return load_legacy_module("behavior", source)


def load_gaze_module() -> ModuleType:
    paths = ProjectPaths.discover()
    source = paths.legacy_monitor_dir / "gaze_detection.py"
    return load_legacy_module("gaze", source)


def load_recognition_module() -> ModuleType:
    paths = ProjectPaths.discover()
    source = paths.legacy_monitor_dir / "face_recognition_system.py"
    return load_legacy_module("recognition", source)
