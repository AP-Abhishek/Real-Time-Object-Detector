from ultralytics import YOLO

def load_model(cfg):
    model_path = cfg["model"]["path"]
    device = cfg["model"].get("device")
    if device:
        return YOLO(model_path).to(device)
    return YOLO(model_path)
