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
            temp_path = f.name
        
        try:
            cfg = load_config(temp_path)
            assert cfg["model"]["device"] == "cpu"
            assert cfg["runtime"]["mode"] == "live"
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_invalid_yaml(self):
        cfg_content = "invalid: yaml: content: ]["
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_empty_config(self):
        cfg_content = ""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(cfg_content)
            f.flush()
            temp_path = f.name
        
        try:
            cfg = load_config(temp_path)
            assert isinstance(cfg, dict)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
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
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
