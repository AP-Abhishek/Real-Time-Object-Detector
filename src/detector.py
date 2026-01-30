from ultralytics import YOLO

def load_model(model_path: str = "yolov8n.pt"):
    return YOLO(model_path)

def detect(model, frame):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        detections.append((x1, y1, x2, y2, label, conf))
    
    return detections