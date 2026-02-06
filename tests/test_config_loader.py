import pytest
from src.config_loader import load_config
from src.validation import ValidationError
import tempfile
import os


class TestLoadConfig:
    
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_valid_config(self):
        cfg_content = """
model:
  path: models/yolov8n.pt
  device: cpu
runtime:
  mode: live
  confidence: 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            
            cfg = load_config(f.name)
            assert cfg["model"]["device"] == "cpu"
            assert cfg["runtime"]["mode"] == "live"
            
            os.unlink(f.name)
    
    def test_invalid_yaml(self):
        cfg_content = "invalid: yaml: content: ]["
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            
            with pytest.raises(ValidationError):
                load_config(f.name)
            
            os.unlink(f.name)
    
    def test_empty_config(self):
        cfg_content = ""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            
            cfg = load_config(f.name)
            assert isinstance(cfg, dict)
            
            os.unlink(f.name)
    
    def test_invalid_config_values(self):
        cfg_content = """
model:
  device: invalid_device
runtime:
  confidence: 1.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            
            with pytest.raises(ValidationError):
                load_config(f.name)
            
            os.unlink(f.name)
