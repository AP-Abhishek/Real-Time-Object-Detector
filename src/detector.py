from ultralytics import YOLO
from pathlib import Path
from src.validation import ValidationError, validate_model_path, validate_frame

def load_model(model_path: str = "models/yolov8n.pt", device: str = "cpu"):
    validate_model_path(model_path)
    
    if device not in ("cpu", "cuda", "mps"):
        raise ValidationError(f"Device must be cpu/cuda/mps")
    
    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    return model

def detect(model, frame, conf_threshold: float = 0.5, allowed_classes: list = None):
    validate_frame(frame)
    
    if not (0 <= conf_threshold <= 1):
        raise ValidationError(f"Confidence must be 0-1")
    
    try:
        results = model(frame, conf=conf_threshold, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    if not results or results[0].boxes is None:
        return []
    
    boxes = results[0].boxes
    names = model.names
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, f"Unknown({cls_id})")

        if allowed_classes is not None and label not in allowed_classes:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append((x1, y1, x2, y2, label, conf))
    
    return detections