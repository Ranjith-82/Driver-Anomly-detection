import unittest

from driver_monitor.monitoring.temporal import ObjectDetectionValidator


class ObjectDetectionValidatorTests(unittest.TestCase):
    @staticmethod
    def detection(
        class_id=1,
        name="Cigarette",
        box=(10, 10, 30, 30),
        confidence=0.8,
    ):
        return {
            "class_id": class_id,
            "name": name,
            "confidence": confidence,
            "box": box,
        }

    def test_single_detection_is_candidate_until_track_persists(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.3,
            min_hits=2,
        )

        first = validator.update(
            [self.detection()], 0.0, frame_shape=(100, 100)
        )
        self.assertEqual(first["confirmed_detections"], [])
        self.assertFalse(first["tracked_detections"][0]["confirmed"])

        early = validator.update(
            [self.detection(box=(11, 10, 31, 30))],
            0.29,
            frame_shape=(100, 100),
        )
        self.assertEqual(early["confirmed_detections"], [])

        confirmed = validator.update(
            [self.detection(box=(11, 10, 31, 30))],
            0.3,
            frame_shape=(100, 100),
        )
        self.assertEqual(len(confirmed["confirmed_detections"]), 1)
        self.assertEqual(confirmed["confirmed_detections"][0]["track_id"], 1)
        self.assertTrue(confirmed["confirmed_detections"][0]["current"])

    def test_unrelated_same_class_object_does_not_inherit_history(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.1,
            min_hits=2,
            max_missed_frames=2,
        )
        validator.update([self.detection()], 0.0, frame_shape=(100, 100))

        new_object = self.detection(box=(60, 60, 80, 80))
        candidate = validator.update(
            [new_object], 0.1, frame_shape=(100, 100)
        )
        self.assertEqual(candidate["confirmed_detections"], [])
        self.assertEqual(candidate["tracked_detections"][0]["track_id"], 2)

        confirmed = validator.update(
            [new_object], 0.2, frame_shape=(100, 100)
        )
        self.assertEqual(
            [item["track_id"] for item in confirmed["confirmed_detections"]],
            [2],
        )

    def test_multiple_same_class_objects_keep_distinct_tracks(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.3,
            min_hits=2,
        )
        left = self.detection(box=(10, 10, 30, 30))
        right = self.detection(box=(60, 60, 80, 80))
        validator.update([left, right], 0.0, frame_shape=(100, 100))

        result = validator.update(
            [
                self.detection(box=(11, 10, 31, 30)),
                self.detection(box=(61, 60, 81, 80)),
            ],
            0.3,
            frame_shape=(100, 100),
        )
        self.assertEqual(
            [item["track_id"] for item in result["confirmed_detections"]],
            [1, 2],
        )

    def test_confirmed_track_survives_misses_then_expires(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.1,
            min_hits=2,
            max_missed_frames=1,
        )
        validator.update([self.detection()], 0.0, frame_shape=(100, 100))
        validator.update([self.detection()], 0.1, frame_shape=(100, 100))

        retained = validator.update([], 0.2, frame_shape=(100, 100))
        self.assertEqual(len(retained["confirmed_detections"]), 1)
        self.assertFalse(retained["confirmed_detections"][0]["current"])

        expired = validator.update([], 0.3, frame_shape=(100, 100))
        self.assertEqual(expired["confirmed_detections"], [])

    def test_stale_track_cannot_be_reused_by_a_later_detection(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.1,
            min_hits=2,
            max_missed_frames=10,
            stale_after_seconds=0.5,
        )
        validator.update([self.detection()], 0.0, frame_shape=(100, 100))
        validator.update([self.detection()], 0.1, frame_shape=(100, 100))

        late = validator.update(
            [self.detection()], 0.7, frame_shape=(100, 100)
        )
        self.assertEqual(late["confirmed_detections"], [])
        self.assertEqual(late["tracked_detections"][0]["track_id"], 2)

    def test_invalid_class_identity_and_boxes_are_ignored(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.0,
            min_hits=2,
        )
        invalid = [
            self.detection(class_id=0, name="Phone"),
            self.detection(name="Cup"),
            self.detection(box=(-1, 10, 30, 30)),
            self.detection(box=(10, 10, 101, 30)),
            self.detection(box=(20, 10, 10, 30)),
        ]

        result = validator.update(
            invalid, 0.0, frame_shape=(100, 100)
        )
        self.assertEqual(result["tracked_detections"], [])
        self.assertEqual(result["confirmed_detections"], [])

    def test_class_changes_do_not_match_an_existing_track(self):
        validator = ObjectDetectionValidator(
            persistence_seconds=0.0,
            min_hits=2,
        )
        validator.update(
            [self.detection(class_id=1, name="Cigarette")],
            0.0,
            frame_shape=(100, 100),
        )
        changed_class = validator.update(
            [self.detection(class_id=2, name="Cup")],
            0.1,
            frame_shape=(100, 100),
        )
        self.assertEqual(changed_class["confirmed_detections"], [])
        self.assertEqual(changed_class["tracked_detections"][0]["track_id"], 2)

    def test_cigarette_cup_and_bottle_can_be_confirmed(self):
        for class_id, name in (
            (1, "Cigarette"),
            (2, "Cup"),
            (3, "Bottle"),
        ):
            with self.subTest(class_name=name):
                validator = ObjectDetectionValidator(
                    persistence_seconds=0.1,
                    min_hits=2,
                )
                detection = self.detection(class_id=class_id, name=name)
                validator.update(
                    [detection], 0.0, frame_shape=(100, 100)
                )
                result = validator.update(
                    [detection], 0.1, frame_shape=(100, 100)
                )
                self.assertEqual(
                    [item["name"] for item in result["confirmed_detections"]],
                    [name],
                )


if __name__ == "__main__":
    unittest.main()
