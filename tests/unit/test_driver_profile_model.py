import unittest

from driver_monitor.profiles.model import DriverProfile


class DriverProfileModelTests(unittest.TestCase):
    def test_user_id_is_required_and_email_is_validated_when_present(self):
        self.assertIn("user_id", DriverProfile("").validate())
        self.assertIn("email", DriverProfile("id", email="invalid").validate())
        self.assertEqual(DriverProfile("id").validate(), {})

    def test_text_fields_and_multiline_address_are_preserved(self):
        profile = DriverProfile(
            "id",
            phone="00123",
            license_number="000007",
            pin_code="00100",
            address="Street one\nApartment two",
        )
        self.assertEqual(profile.phone, "00123")
        self.assertEqual(profile.license_number, "000007")
        self.assertEqual(profile.pin_code, "00100")
        self.assertIn("\n", profile.address)

    def test_legacy_date_and_photo_fields_are_supported(self):
        profile = DriverProfile.from_mapping({
            "user_id": "id",
            "dob": "2000-01-01",
            "photo": "driver_photos/id.jpg",
        })
        self.assertEqual(profile.date_of_birth, "2000-01-01")
        self.assertEqual(profile.photo_reference, "driver_photos/id.jpg")
        self.assertEqual(profile.to_legacy_mapping()["dob"], "2000-01-01")


if __name__ == "__main__":
    unittest.main()
