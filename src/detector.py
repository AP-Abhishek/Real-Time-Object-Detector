from ultralytics import YOLO

def load_model(model_path: str = "yolov8n.pt"):
    return YOLO(model_path)

def detect(model, frame, conf_threshold: float = 0.5, allowed_classes: list = None):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if allowed_classes is not None and label not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2, label, conf))
    
    return detections