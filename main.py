import argparse
import cv2
import time
import torch

from src.camera import open_camera, read_frame, release_camera
from src.detector import load_model, detect
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, CAMERA_INDEX, WINDOW_NAME
from src.logger import setup_logger
from src.tracker import CentroidTracker
from src.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection Application")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Webcam index (default from config)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (overrides webcam)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold (default from config)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names to detect (overrides config)",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Limit processing FPS (e.g. 15). Unlimited if omitted.",
    )
    return parser.parse_args()

def parse_classes(value):
    if not value:
        return ALLOWED_CLASSES
    return [c.strip() for c in value.split(",") if c.strip()]

def main():
    args = parse_args()
    logger = setup_logger()
    tracker = CentroidTracker()

    allowed_classes = parse_classes(args.classes)
    frame_interval = 1.0 / args.max_fps if args.max_fps and args.max_fps > 0 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Confidence threshold: {args.conf}")
    if frame_interval:
        logger.info(f"FPS limited to: {args.max_fps}")

    if args.video:
        logger.info(f"Using video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error("Failed to open video file.")
            return
        release_fn = cap.release
    else:
        logger.info(f"Using webcam index: {args.camera}")
        try:
            cap = open_camera(args.camera)
        except RuntimeError as e:
            logger.error(str(e))
            return
        release_fn = lambda: release_camera(cap)
    
    try:
        model = load_model(MODEL_PATH, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        release_fn()
        return
    
    logger.info("Model loaded successfully.")

    run_pipeline(
        cap=cap,
        model=model,
        logger=logger,
        conf=args.conf,
        allowed_classes=allowed_classes,
        max_fps=args.max_fps,
        is_video=bool(args.video),
    )

    release_fn()
    cv2.destroyAllWindows()
    logger.info("Application terminated gracefully.")

if __name__ == "__main__":
    main()
