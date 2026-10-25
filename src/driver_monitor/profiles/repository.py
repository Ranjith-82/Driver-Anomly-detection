from __future__ import annotations

import os
import pickle
import tempfile
import threading
from pathlib import Path

from driver_monitor.profiles.model import DriverProfile


class DriverProfileRepository:
    """Preserves the existing pickle mapping format for local profile compatibility."""

    def __init__(
        self,
        database_path: Path,
        managed_photo_dir: Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.managed_photo_dir = (
            Path(managed_photo_dir) if managed_photo_dir is not None else None
        )
        self._lock = threading.RLock()

    def _load_records(self) -> dict[str, dict[str, object]]:
        if not self.database_path.exists():
            return {}
        with self.database_path.open("rb") as file:
            records = pickle.load(file)
        if not isinstance(records, dict):
            raise ValueError("Driver database must contain a mapping of IDs to profiles")
        normalized: dict[str, dict[str, object]] = {}
        for key, value in records.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("Driver database contains an invalid profile record")
            normalized[key] = value
        return normalized

    def _save_records(self, records: dict[str, dict[str, object]]) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.database_path.parent,
                prefix=f".{self.database_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                pickle.dump(records, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.database_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def list_profiles(self) -> list[DriverProfile]:
        with self._lock:
            return [
                DriverProfile.from_mapping(value)
                for _, value in sorted(self._load_records().items())
            ]

    def get(self, user_id: str) -> DriverProfile | None:
        with self._lock:
            record = self._load_records().get(user_id)
            return DriverProfile.from_mapping(record) if record is not None else None

    def search(self, query: str) -> list[DriverProfile]:
        term = query.strip().casefold()
        profiles = self.list_profiles()
        if not term:
            return profiles
        return [
            profile for profile in profiles
            if term in profile.user_id.casefold()
            or term in profile.full_name.casefold()
            or term in profile.username.casefold()
            or term in profile.email.casefold()
        ]

    def create(self, profile: DriverProfile) -> None:
        self._raise_validation_errors(profile)
        with self._lock:
            records = self._load_records()
            if profile.user_id in records:
                raise ValueError(f"User ID already exists: {profile.user_id}")
            records[profile.user_id] = profile.to_legacy_mapping()
            self._save_records(records)

    def update(self, profile: DriverProfile) -> None:
        self._raise_validation_errors(profile)
        with self._lock:
            records = self._load_records()
            if profile.user_id not in records:
                raise KeyError(f"Unknown user ID: {profile.user_id}")
            existing = records[profile.user_id]
            updated = profile.to_legacy_mapping()
            if not updated.get("photo"):
                updated["photo"] = existing.get(
                    "photo", existing.get("photo_reference", "")
                )
            existing.update(updated)
            records[profile.user_id] = existing
            self._save_records(records)

    def delete(
        self,
        user_id: str,
        *,
        confirm: bool = False,
        remove_managed_photo: bool = False,
    ) -> None:
        if not confirm:
            raise PermissionError("Profile deletion requires explicit confirmation")
        with self._lock:
            records = self._load_records()
            record = records.pop(user_id, None)
            if record is None:
                raise KeyError(f"Unknown user ID: {user_id}")
            self._save_records(records)
        if remove_managed_photo and self.managed_photo_dir is not None:
            photo_reference = record.get("photo_reference", record.get("photo"))
            if isinstance(photo_reference, str) and photo_reference:
                photo_path = Path(photo_reference)
                if not photo_path.is_absolute():
                    photo_path = self.database_path.parent / photo_path
                managed_root = self.managed_photo_dir.resolve()
                resolved_photo = photo_path.resolve()
                if resolved_photo.is_relative_to(managed_root) and resolved_photo.is_file():
                    resolved_photo.unlink()

    @staticmethod
    def _raise_validation_errors(profile: DriverProfile) -> None:
        errors = profile.validate()
        if errors:
            raise ValueError("; ".join(errors.values()))
