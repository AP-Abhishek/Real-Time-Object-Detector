from src.pipeline import run_pipeline
from src.config_loader import load_config
from src.model import load_model
from src.camera import open_camera, open_video

def main():
    cfg = load_config()

    runtime = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})

    mode = runtime.get("mode", "live")
    conf = runtime.get("confidence", 0.5)
    max_fps = runtime.get("max_fps")
    window_name = runtime.get("window_name", "Detection")
    output_dir = runtime.get("output_dir") or "runs/latest"

    allowed_classes = runtime.get("allowed_classes")
    if isinstance(allowed_classes, (list, tuple)):
        allowed_classes = set(map(int, allowed_classes))
    else:
        allowed_classes = None

    model = load_model(cfg)

    is_video = mode == "video"
    headless = mode in ("headless", "benchmark")
    benchmark = mode == "benchmark"

    if mode == "video":
        cap = open_video(runtime.get("video_path"))
    else:
        cap = open_camera(runtime.get("camera_index", 0))

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
        benchmark=benchmark,
    )

if __name__ == "__main__":
    main()
