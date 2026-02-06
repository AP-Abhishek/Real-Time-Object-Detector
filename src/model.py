from ultralytics import YOLO
from src.validation import ValidationError, validate_model_path

def load_model(cfg):
    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", "models/yolov8n.pt")
    device = model_cfg.get("device", "cpu")
    
    validate_model_path(model_path)
    
    try:
        model = YOLO(model_path)
        if device:
            model.to(device)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    return model

