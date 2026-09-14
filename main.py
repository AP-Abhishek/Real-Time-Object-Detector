import os
os.environ["YOLO_VERBOSE"] = "False"
import sys
import argparse
from src.pipeline import run_pipeline
from src.config_loader import load_config
from src.model import load_model
from src.camera import open_camera, open_video
from src.logger import setup_logger, get_logger
from src.validation import ValidationError
from src.device_utils import get_device_info, validate_and_fallback

class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=42, width=110)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using YOLOv8",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:
  rtod                                    # Webcam live detection
  rtod --video video.mp4                  # Video file detection
  rtod --confidence 0.7                   # Higher confidence threshold
  rtod --headless                         # No display window
  rtod --model models/yolov8s.pt          # Small model
        """
    )

    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path (default: config.yaml)")
    parser.add_argument("--mode", type=str, choices=["live", "video", "headless", "benchmark"],
                        help="Execution mode")
    parser.add_argument("--confidence", type=float, metavar="[0-1]",
                        help="Detection confidence threshold")
    parser.add_argument("--camera", type=int, metavar="INDEX",
                        help="Camera device index")
    parser.add_argument("--video", type=str, metavar="PATH",
                        help="Video file path (sets mode to video)")
    parser.add_argument("--output", type=str, metavar="DIR",
                        help="Output directory")
    parser.add_argument("--model", type=str, metavar="PATH",
                        help="Model file path")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                        help="Compute device")
    parser.add_argument("--max-fps", type=float, metavar="N",
                        help="Maximum FPS limit")
    parser.add_argument("--headless", action="store_true",
                        help="No display window (headless mode)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode (no display, print metrics)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated video output to file")
    parser.add_argument("-v", "--version", action="version", version="1.0.0")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    logger = setup_logger()
    
    dev_info = get_device_info()
    if dev_info["cuda_available"]:
        logger.info(f"GPU available: {dev_info['cuda_device_name']}")
    else:
        logger.info("GPU not available, using CPU")
    
    try:
        logger.info("Loading configuration...")
        cfg = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Config not found: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Config invalid: {e}")
        sys.exit(1)

    runtime = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})

    if args.mode:
        runtime["mode"] = args.mode
    if args.video:
        runtime["mode"] = "video"
        runtime["video_path"] = args.video
    if args.confidence is not None:
        runtime["confidence"] = args.confidence
    if args.camera is not None:
        runtime["camera_index"] = args.camera
    if args.output:
        runtime["output_dir"] = args.output
    if args.max_fps is not None:
        runtime["max_fps"] = args.max_fps
    if args.headless:
        runtime["mode"] = "headless"
    if args.benchmark:
        runtime["mode"] = "benchmark"
    if args.save_video:
        runtime["save_video"] = True
    if args.model:
        model_cfg["path"] = args.model
    if args.device:
        model_cfg["device"] = args.device

    mode = runtime.get("mode", "live")
    conf = runtime.get("confidence", 0.5)
    max_fps = runtime.get("max_fps")
    window_name = runtime.get("window_name", "Detection")
    output_dir = runtime.get("output_dir") or "runs/latest"
    save_video = runtime.get("save_video", False)

    logger.info(f"Mode: {mode}, Confidence: {conf}, FPS: {max_fps}")

    allowed_classes = runtime.get("allowed_classes")
    if isinstance(allowed_classes, (list, tuple, set)):
        allowed_classes = set(allowed_classes)
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
            save_video=save_video,
        )
        logger.info("Pipeline completed successfully")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received (Ctrl+C). Exiting pipeline cleanly.")
        sys.exit(0)
    except (ValidationError, RuntimeError) as e:
        logger.error(f"Failed to open camera/video: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
