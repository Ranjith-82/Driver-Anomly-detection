from __future__ import annotations

import re
from dataclasses import asdict, dataclass


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class DriverProfile:
    user_id: str
    full_name: str = ""
    username: str = ""
    email: str = ""
    phone: str = ""
    blood_group: str = ""
    license_number: str = ""
    date_of_birth: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    pin_code: str = ""
    photo_reference: str = ""
    created_date: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "DriverProfile":
        values = {
            field: str(data.get(field, "") or "")
            for field in (
                "user_id",
                "full_name",
                "username",
                "email",
                "phone",
                "blood_group",
                "license_number",
                "address",
                "city",
                "state",
                "pin_code",
                "created_date",
            )
        }
        values["date_of_birth"] = str(
            data.get("date_of_birth", data.get("dob", "")) or ""
        )
        values["photo_reference"] = str(
            data.get("photo_reference", data.get("photo", "")) or ""
        )
        return cls(**values)

    def validate(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        if not self.user_id.strip():
            errors["user_id"] = "User ID is required."
        if self.email and not EMAIL_PATTERN.fullmatch(self.email.strip()):
            errors["email"] = "Enter a valid email address."
        return errors

    def to_legacy_mapping(self) -> dict[str, str]:
        values = asdict(self)
        values["dob"] = values.pop("date_of_birth")
        values["photo"] = values.pop("photo_reference")
        return values
