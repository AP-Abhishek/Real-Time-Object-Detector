import pytest
from src.validation import validate_config, ValidationError, validate_model_path, validate_video_path
from pathlib import Path
import tempfile


class TestValidateConfig:
    
    def test_valid_config(self):
        cfg = {
            "model": {"path": "models/yolov8n.pt", "device": "cpu"},
            "runtime": {"mode": "live", "confidence": 0.5, "camera_index": 0}
        }
        result = validate_config(cfg)
        assert result["model"]["device"] == "cpu"
        assert result["runtime"]["mode"] == "live"
    
    def test_invalid_device(self):
        cfg = {
            "model": {"device": "invalid"},
            "runtime": {}
        }
        with pytest.raises(ValidationError):
            validate_config(cfg)
    
    def test_invalid_mode(self):
        cfg = {
            "model": {},
            "runtime": {"mode": "invalid"}
        }
        with pytest.raises(ValidationError):
            validate_config(cfg)
    
    def test_invalid_confidence(self):
        cfg = {
            "model": {},
            "runtime": {"confidence": 1.5}
        }
        with pytest.raises(ValidationError):
            validate_config(cfg)
    
    def test_video_mode_needs_path(self):
        cfg = {
            "model": {},
            "runtime": {"mode": "video"}
        }
        with pytest.raises(ValidationError):
            validate_config(cfg)
    
    def test_invalid_camera_index(self):
        cfg = {
            "model": {},
            "runtime": {"camera_index": -1}
        }
        with pytest.raises(ValidationError):
            validate_config(cfg)


class TestValidateModelPath:
    
    def test_nonexistent_file(self):
        with pytest.raises(ValidationError):
            validate_model_path("/nonexistent/model.pt")
    
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = validate_model_path(f.name)
            assert isinstance(path, Path)
            assert path.exists()


class TestValidateVideoPath:
    
    def test_nonexistent_file(self):
        with pytest.raises(ValidationError):
            validate_video_path("/nonexistent/video.mp4")
    
    def test_invalid_format(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            with pytest.raises(ValidationError):
                validate_video_path(f.name)
    
    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = validate_video_path(f.name)
            assert isinstance(path, Path)
            assert path.exists()
