import unittest

from driver_monitor.monitoring.metrics import LatencyTracker


class LatencyMetricsTests(unittest.TestCase):
    def test_counts_only_changed_combined_results(self):
        tracker = LatencyTracker(target_seconds=2)
        self.assertTrue(tracker.record(1.0, 2.0, ("normal", "uncertain")))
        self.assertFalse(tracker.record(2.0, 3.0, ("normal", "uncertain")))
        self.assertTrue(tracker.record(3.0, 5.5, ("distracted", "uncertain")))
        summary = tracker.summary()
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["within_target"], 1)
        self.assertEqual(summary["fraction_within_target"], 0.5)

    def test_rejects_non_monotonic_timestamps_and_clears_samples(self):
        tracker = LatencyTracker()
        with self.assertRaises(ValueError):
            tracker.record(2.0, 1.0, "result")
        tracker.record(1.0, 1.5, "result")
        tracker.clear()
        self.assertEqual(tracker.summary()["samples"], 0)


if __name__ == "__main__":
    unittest.main()
