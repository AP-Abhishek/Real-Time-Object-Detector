import argparse
import cv2
import time
import torch

from src.camera import open_camera, read_frame, release_camera
from src.detector import load_model, detect
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, CAMERA_INDEX, WINDOW_NAME
from src.logger import setup_logger

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
    return parser.parse_args()

def parse_classes(value):
    if not value:
        return ALLOWED_CLASSES
    return [c.strip() for c in value.split(",") if c.strip()]

def main():
    args = parse_args()
    logger = setup_logger()

    allowed_classes = parse_classes(args.classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Confidence threshold: {args.conf}")

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
        release_camera(cap)
        release_fn()
        return
    
    logger.info("Model loaded successfully.")

    prev_time = 0.0
    running = True

    while running:
        ret, frame = cap.read() if args.video else read_frame(cap)
        if not ret:
            logger.warning("Failed to read frame from camera.")
            break

        detections = detect(model, frame, args.conf, allowed_classes)
        
        for x1, y1, x2, y2, label, conf in detections:
            text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            running = False

    release_camera(cap)
    cv2.destroyAllWindows()
    logger.info("Application terminated gracefully.")

if __name__ == "__main__":
    main()
