import cv2
from ultralytics import YOLO

from src.config import Config
from src.pipeline import run_pipeline

def main():
    cfg = Config("config.yaml")

    model_path = cfg.get("model", "path")
    conf = cfg.get("model", "conf")
    allowed_classes = cfg.get("model", "allowed_classes")

    source = cfg.get("input", "source")
    is_video = cfg.get("input", "is_video")
    max_fps = cfg.get("input", "max_fps")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    window_name = cfg.get("runtime", "window_name")
    output_dir = cfg.get("runtime", "output_dir")

    run_pipeline(
        cap=cap,
        model=model,
        logger=None,
        conf=conf,
        allowed_classes=allowed_classes,
        window_name=window_name,
        output_dir=output_dir,
        max_fps=max_fps,
        is_video=is_video,
    )

if __name__ == "__main__":
    main()
