from ultralytics import YOLO
from pathlib import Path

def load_model(model_path: str = "models/yolov8n.pt", device: str = "cpu"):
    model_path = Path(model_path)
    model = YOLO(model_path)
    model.to(device)
    return model

def detect(model, frame, conf_threshold: float = 0.5, allowed_classes: list = None):
    results = model(frame, conf=conf_threshold, verbose=False)

    if not results or results[0].boxes is None:
        return []
    
    boxes = results[0].boxes
    names = model.names
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id)

        if allowed_classes is not None and label not in allowed_classes:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append((x1, y1, x2, y2, label, conf))
    
    return detections