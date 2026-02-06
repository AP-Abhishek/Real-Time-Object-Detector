import time
import cv2
import logging
from pathlib import Path
from typing import Optional, Set

from src.camera import read_frame, release_camera
from src.detector import detect
from src.tracker import CentroidTracker
from src.exporter import export_exit_stats_csv
from src.logger import get_logger
from src.validation import ValidationError

def run_pipeline(
    cap: cv2.VideoCapture,
    model,
    logger: Optional[logging.Logger],
    conf: float,
    allowed_classes: Optional[Set],
    window_name: str,
    output_dir: str,
    max_fps: Optional[float] = None,
    is_video: bool = False,
    headless: bool = False,
    benchmark: bool = False,
) -> None:
    if logger is None:
        logger = get_logger()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    tracker = CentroidTracker()

    frame_interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
    prev_time = 0.0
    last_frame_time = 0.0
    frames = 0
    start_time = time.time()
    running = True

    logger.info(f"Starting frame processing (headless={headless}, benchmark={benchmark})")

    try:
        while running:
            now = time.time()
            if frame_interval and (now - last_frame_time) < frame_interval:
                time.sleep(frame_interval - (now - last_frame_time))
            last_frame_time = time.time()

            ret, frame = cap.read() if is_video else read_frame(cap)
            if not ret or frame is None:
                logger.info("No more frames available")
                break

            try:
                detections = detect(model, frame, conf, allowed_classes)
            except (ValidationError, RuntimeError) as e:
                logger.warning(f"Detection error on frame {frames}: {e}")
                continue

            rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections]
            objects, _ = tracker.update(rects)

            if not headless:
                try:
                    for (x1, y1, x2, y2, label, score) in detections:
                        cX = int((x1 + x2) / 2)
                        cY = int((y1 + y2) / 2)

                        object_id = None
                        min_dist = float("inf")
                        for oid, (oX, oY) in objects.items():
                            d = (cX - oX) ** 2 + (cY - oY) ** 2
                            if d < min_dist:
                                min_dist = d
                                object_id = oid

                        lifetime = int(time.time() - tracker.start_time.get(object_id, time.time()))
                        text = f"ID {object_id} | {label} {score:.2f} | {lifetime}s"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    current_time = time.time()
                    fps = 1 / (current_time - prev_time) if prev_time else 0
                    prev_time = current_time

                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("User quit signal received")
                        running = False
                except Exception as e:
                    logger.warning(f"Rendering error on frame {frames}: {e}")
                    continue

            frames += 1

    finally:
        end_time = time.time()
        logger.info("Cleaning up resources...")
        release_camera(cap)
        if not headless:
            cv2.destroyAllWindows()

        try:
            export_exit_stats_csv(tracker.exit_stats, output_dir)
            logger.info(f"Exported {len(tracker.exit_stats)} object statistics")
        except Exception as e:
            logger.error(f"Export failed: {e}")

        if benchmark:
            duration = end_time - start_time
            fps = frames / duration if duration > 0 else 0
            logger.info(f"Benchmark - Frames: {frames}, Duration: {duration:.2f}s, Avg FPS: {fps:.2f}")


