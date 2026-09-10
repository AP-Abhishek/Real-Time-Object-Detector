from typing import Dict, Any, Union
from ultralytics import YOLO
from src.validation import ValidationError, validate_model_path
from src.device_utils import validate_and_fallback

def load_model(cfg_or_path: Union[Dict[str, Any], str] = "models/yolov8n.pt", device: str = "cpu") -> YOLO:
    if isinstance(cfg_or_path, dict):
        model_cfg = cfg_or_path.get("model", {})
        model_path = model_cfg.get("path", "models/yolov8n.pt")
        target_device = model_cfg.get("device", "cpu")
    else:
        model_path = cfg_or_path
        target_device = device
    
    validate_model_path(model_path)
    final_device = validate_and_fallback(target_device)
    
    try:
        model = YOLO(model_path)
        if final_device:
            model.to(final_device)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    return model

