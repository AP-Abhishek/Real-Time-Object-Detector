import pytest
import numpy as np
from src.tracker import CentroidTracker


class TestCentroidTracker:

    def test_tracker_registration_and_update(self):
        tracker = CentroidTracker(max_disappeared=5)

        rects = [(10, 20, 50, 60, "person", 0.90)]
        objects, events = tracker.update(rects)

        assert len(objects) == 1
        assert 0 in objects
        assert events["entered"] == [0]

        info = tracker.get_tracked_info()
        assert 0 in info
        assert info[0]["label"] == "person"
        assert info[0]["box"] == (10, 20, 50, 60)

    def test_tracker_disappearance_and_deregistration(self):
        tracker = CentroidTracker(max_disappeared=2)

        rects = [(10, 20, 50, 60, "person", 0.90)]
        tracker.update(rects)

        tracker.update([])
        tracker.update([])
        _, events = tracker.update([])

        assert 0 in events["exited"]
        assert 0 in tracker.exit_stats
        assert tracker.exit_stats[0][1] == 1

    def test_deregister_all(self):
        tracker = CentroidTracker()
        rects = [
            (10, 20, 50, 60, "person", 0.90),
            (100, 100, 150, 150, "car", 0.85),
        ]
        tracker.update(rects)

        tracker.deregister_all()
        assert len(tracker.objects) == 0
        assert 0 in tracker.exit_stats
        assert 1 in tracker.exit_stats
