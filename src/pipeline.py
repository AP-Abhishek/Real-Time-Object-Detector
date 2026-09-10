import time
import cv2
import logging
from pathlib import Path
from typing import Optional, Set

from src.camera import read_frame, release_camera
from src.detector import detect
from src.tracker import CentroidTracker
from src.runtime_tracker import RuntimeTracker
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

    device_name = "cuda" if hasattr(model, "device") and "cuda" in str(model.device) else ("mps" if hasattr(model, "device") and "mps" in str(model.device) else "cpu")
    runtime_tracker = None

    frame_interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
    prev_time = 0.0
    last_frame_time = 0.0
    frames = 0
    fps_samples = []
    resolution = None
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

            if resolution is None and frame is not None:
                resolution = (frame.shape[1], frame.shape[0])

            if runtime_tracker is None:
                runtime_tracker = RuntimeTracker(
                    model=model,
                    device=device_name,
                    confidence_threshold=conf,
                    input_type="video" if is_video else "camera",
                    resolution=resolution or (640, 480),
                    output_root=output_dir,
                )

            try:
                detections = detect(model, frame, conf, allowed_classes)
            except (ValidationError, RuntimeError) as e:
                logger.warning(f"Detection error on frame {frames}: {e}")
                continue

            _, events = tracker.update(detections)

            for oid in events.get("entered", []):
                runtime_tracker.register_object(oid, tracker.labels.get(oid, "object"))

            for oid in tracker.get_tracked_info():
                runtime_tracker.update_object(oid)

            for oid in events.get("exited", []):
                runtime_tracker.unregister_object(oid)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time
            if fps > 0:
                fps_samples.append(fps)
                runtime_tracker.tick_frame(fps)

            if not headless:
                try:
                    tracked_info = tracker.get_tracked_info()
                    for oid, obj in tracked_info.items():
                        x1, y1, x2, y2 = obj["box"]
                        label = obj["label"]
                        score = obj["score"]
                        lifetime = int(time.time() - obj["start_time"])
                        text = f"ID {oid} | {label} {score:.2f} | {lifetime}s"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            text,
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

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
            tracker.deregister_all()
            if runtime_tracker:
                runtime_tracker.unregister_all()
                runtime_tracker.finalize()
                logger.info(f"Exported run metadata, index, and summary statistics to {output_dir}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

        if benchmark:
            duration = end_time - start_time
            fps = frames / duration if duration > 0 else 0
            logger.info(f"Benchmark - Frames: {frames}, Duration: {duration:.2f}s, Avg FPS: {fps:.2f}")



