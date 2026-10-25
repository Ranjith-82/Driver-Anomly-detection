from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    repository_root: Path
    legacy_monitor_dir: Path
    legacy_profiles_dir: Path
    local_data_dir: Path
    behavior_model: Path
    behavior_labels: Path
    face_landmarker: Path
    object_model: Path

    @classmethod
    def discover(cls, repository_root: Path | None = None) -> "ProjectPaths":
        root = (repository_root or REPOSITORY_ROOT).resolve()
        monitor_dir = (
            root
            / "Phase-3"
            / "Buffered-system"
            / "buffered_system"
            / "Buffer"
            / "Buffered_system"
        )
        profiles_dir = root / "Phase-2" / "Face_recognition_system"
        models_dir = root / "models"
        return cls(
            repository_root=root,
            legacy_monitor_dir=monitor_dir,
            legacy_profiles_dir=profiles_dir,
            local_data_dir=root / "data" / "local",
            behavior_model=models_dir / "behavior" / "driver_behavior_model.pkl",
            behavior_labels=models_dir / "behavior" / "driver_behavior_labels.pkl",
            face_landmarker=models_dir / "face_landmarker" / "face_landmarker.task",
            object_model=models_dir / "objects" / "best_phone_finetuned_v1.pt",
        )

    @property
    def driver_database(self) -> Path:
        legacy_database = self.legacy_profiles_dir / "drivers_database.pkl"
        if legacy_database.exists():
            return legacy_database
        return self.local_data_dir / "drivers_database.pkl"

    @property
    def driver_photos(self) -> Path:
        if self.driver_database.parent == self.legacy_profiles_dir:
            return self.legacy_profiles_dir / "driver_photos"
        return self.local_data_dir / "driver_photos"

    @property
    def emotion_thresholds(self) -> Path:
        return self.local_data_dir / "emotion-thresholds.json"
