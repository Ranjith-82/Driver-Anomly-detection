from __future__ import annotations

from driver_monitor.profiles.model import DriverProfile
from driver_monitor.profiles.repository import DriverProfileRepository


class DriverProfileController:
    def __init__(self, repository: DriverProfileRepository) -> None:
        self.repository = repository

    def save(self, profile: DriverProfile, *, is_new: bool) -> None:
        if is_new:
            self.repository.create(profile)
        else:
            self.repository.update(profile)

    def search(self, text: str) -> list[DriverProfile]:
        return self.repository.search(text)

    def delete(self, user_id: str, *, confirmed: bool) -> None:
        self.repository.delete(
            user_id,
            confirm=confirmed,
            remove_managed_photo=True,
        )
