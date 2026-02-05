from src.pipeline import run_pipeline
from src.config_loader import load_config
from src.model import load_model
from src.camera import open_camera, open_video

def main():
    cfg = load_config()

    mode = cfg.get("runtime", "mode") or "live"
    conf = cfg.get("runtime", "confidence")
    allowed_classes = cfg.get("runtime", "allowed_classes")
    max_fps = cfg.get("runtime", "max_fps")
    window_name = cfg.get("runtime", "window_name") or "Detection"
    output_dir = cfg.get("runtime", "output_dir") or "runs/latest"

    if allowed_classes:
        allowed_classes = set(map(int, allowed_classes))
    else:
        allowed_classes = None

    model = load_model(cfg)

    is_video = mode == "video"
    headless = mode in ("headless", "benchmark")

    if mode == "video":
        cap = open_video(cfg.get("runtime", "video_path"))
    else:
        cap = open_camera(cfg.get("runtime", "camera_index"))

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
        headless=headless,
        benchmark=(mode == "benchmark"),
    )

if __name__ == "__main__":
    main()
