import pytest
import numpy as np
from unittest.mock import MagicMock
from src.detector import detect
from src.validation import ValidationError


class TestDetector:

    def test_invalid_frame(self):
        mock_model = MagicMock()
        with pytest.raises(ValidationError):
            detect(mock_model, None)

        with pytest.raises(ValidationError):
            detect(mock_model, np.zeros((100, 100), dtype=np.uint8))

    def test_invalid_confidence_threshold(self):
        mock_model = MagicMock()
        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValidationError):
            detect(mock_model, valid_frame, conf_threshold=-0.1)

        with pytest.raises(ValidationError):
            detect(mock_model, valid_frame, conf_threshold=1.5)

    def test_empty_results(self):
        mock_model = MagicMock()
        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_results = MagicMock()
        mock_results.boxes = None
        mock_model.return_value = [mock_results]

        detections = detect(mock_model, valid_frame)
        assert detections == []

    def test_detection_parsing_and_allowed_classes(self):
        valid_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "car", 2: "dog"}

        box1 = MagicMock()
        box1.cls = [0]
        box1.conf = [0.85]
        box1.xyxy = [[10, 20, 50, 60]]

        box2 = MagicMock()
        box2.cls = [1]
        box2.conf = [0.90]
        box2.xyxy = [[100, 120, 200, 220]]

        box3 = MagicMock()
        box3.cls = [2]
        box3.conf = [0.75]
        box3.xyxy = [[300, 300, 400, 400]]

        mock_results = MagicMock()
        mock_results.boxes = [box1, box2, box3]
        mock_model.return_value = [mock_results]

        detections = detect(mock_model, valid_frame, conf_threshold=0.5)
        assert len(detections) == 3
        assert detections[0] == (10, 20, 50, 60, "person", 0.85)

        detections_filtered = detect(
            mock_model,
            valid_frame,
            conf_threshold=0.5,
            allowed_classes={"person", "car"},
        )
        assert len(detections_filtered) == 2
        labels = [d[4] for d in detections_filtered]
        assert "person" in labels and "car" in labels

        detections_int_filtered = detect(
            mock_model, valid_frame, conf_threshold=0.5, allowed_classes={0}
        )
        assert len(detections_int_filtered) == 1
        assert detections_int_filtered[0][4] == "person"

