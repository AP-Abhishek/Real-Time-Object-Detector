import os
import json
import csv
import tempfile
import pytest
from unittest.mock import MagicMock
from src.exporter import export_exit_stats_csv, export_run_json


class TestExporter:

    def test_export_exit_stats_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_stats = {0: (2.5, 30), 1: (5.0, 60)}
            export_exit_stats_csv(exit_stats, tmpdir)

            csv_path = os.path.join(tmpdir, "summary.csv")
            assert os.path.exists(csv_path)

            with open(csv_path, "r", newline="") as f:
                reader = list(csv.reader(f))
                assert len(reader) == 3
                assert reader[0] == [
                    "object_id",
                    "lifetime_seconds",
                    "total_frames",
                    "exit_timestamp",
                ]
                assert reader[1][0] == "0"
                assert reader[1][1] == "2.5"
                assert reader[1][2] == "30"

    def test_export_run_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.model_name = "yolov8n.pt"

            export_run_json(
                run_id="run_1234",
                start_time="2026-09-11 10:00:00",
                end_time="2026-09-11 10:01:00",
                duration_seconds=60.0,
                model=mock_model,
                device="cpu",
                confidence_threshold=0.5,
                input_type="camera",
                resolution=(640, 480),
                total_frames=1800,
                fps_samples=[30.0, 30.0],
                objects={
                    0: {
                        "class": "person",
                        "first_seen": "2026-09-11 10:00:01",
                        "last_seen": "2026-09-11 10:00:05",
                        "lifetime_seconds": 4.0,
                        "total_frames": 120,
                    }
                },
                class_aggregates={
                    "person": {
                        "count": 1,
                        "total_lifetime": 4.0,
                        "total_frames": 120,
                        "avg_lifetime": 4.0,
                        "avg_frames": 120,
                    }
                },
                output_dir=tmpdir,
            )

            json_path = os.path.join(tmpdir, "summary.json")
            assert os.path.exists(json_path)

            with open(json_path, "r") as f:
                data = json.load(f)
                assert data["run_metadata"]["run_id"] == "run_1234"
                assert data["run_metadata"]["device"] == "cpu"
                assert "0" in data["objects"] or 0 in data["objects"]
                assert "person" in data["class_aggregates"]
