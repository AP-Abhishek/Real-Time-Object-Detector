from typing import Dict, Any
from ultralytics import YOLO
from src.validation import ValidationError, validate_model_path
from src.device_utils import validate_and_fallback

def load_model(cfg: Dict[str, Any]) -> YOLO:
    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", "models/yolov8n.pt")
    device = model_cfg.get("device", "cpu")
    
    validate_model_path(model_path)
    
    final_device = validate_and_fallback(device)
    
    try:
        model = YOLO(model_path)
        if final_device:
            model.to(final_device)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    return model

