from ultralytics import YOLO

def load_model(cfg):
    model_cfg = cfg.get("model", {})
    model_path = model_cfg["path"]
    device = model_cfg.get("device")
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model
