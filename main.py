from src.pipeline import run_pipeline
from src.config_loader import load_config
from src.model import load_model
from src.camera import open_camera, open_video
from src.logger import setup_logger, get_logger
from src.validation import ValidationError
import sys

def main():
    logger = setup_logger()
    
    try:
        logger.info("Loading configuration...")
        cfg = load_config()
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Config not found: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Config invalid: {e}")
        sys.exit(1)

    runtime = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})

    mode = runtime.get("mode", "live")
    conf = runtime.get("confidence", 0.5)
    max_fps = runtime.get("max_fps")
    window_name = runtime.get("window_name", "Detection")
    output_dir = runtime.get("output_dir") or "runs/latest"

    logger.info(f"Mode: {mode}, Confidence: {conf}, FPS: {max_fps}")

    allowed_classes = runtime.get("allowed_classes")
    if isinstance(allowed_classes, (list, tuple)):
        allowed_classes = set(map(int, allowed_classes))
    else:
        allowed_classes = None

    try:
        logger.info(f"Loading model from {model_cfg.get('path')}...")
        model = load_model(cfg)
        logger.info(f"Model loaded on device: {model_cfg.get('device')}")
    except (ValidationError, RuntimeError) as e:
        logger.error(f"Model load failed: {e}")
        sys.exit(1)

    is_video = mode == "video"
    headless = mode in ("headless", "benchmark")
    benchmark = mode == "benchmark"

    try:
        if mode == "video":
            logger.info(f"Opening video: {runtime.get('video_path')}")
            cap = open_video(runtime.get("video_path"))
        else:
            logger.info(f"Opening camera at index {runtime.get('camera_index', 0)}")
            cap = open_camera(runtime.get("camera_index", 0))
        logger.info("Camera/video opened successfully")
    except (ValidationError, RuntimeError) as e:
        logger.error(f"Failed to open camera/video: {e}")
        sys.exit(1)

    try:
        logger.info("Starting detection pipeline...")
        run_pipeline(
            cap=cap,
            model=model,
            logger=logger,
            conf=conf,
            allowed_classes=allowed_classes,
            window_name=window_name,
            output_dir=output_dir,
            max_fps=max_fps,
            is_video=is_video,
            headless=headless,
            benchmark=benchmark,
        )
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


