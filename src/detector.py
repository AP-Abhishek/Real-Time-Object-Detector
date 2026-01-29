from ultralytics import YOLO

def load_model(model_path: str = "yolov8n.pt"):
    return YOLO(model_path)

def detect(model, frame):
    results = model(frame, verbose=False)
    return results[0].plot()