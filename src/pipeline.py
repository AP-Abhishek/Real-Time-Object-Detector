import time
import cv2
import logging
from pathlib import Path
from typing import Optional, Set

from src.camera import read_frame, release_camera
from src.detector import detect
from src.tracker import CentroidTracker
from src.exporter import export_exit_stats_csv, export_run_json
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
    fps_samples = []
    resolution = None
    objects_dict = {}
    class_aggregates = {}
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

            try:
                detections = detect(model, frame, conf, allowed_classes)
            except (ValidationError, RuntimeError) as e:
                logger.warning(f"Detection error on frame {frames}: {e}")
                continue

            for x1, y1, x2, y2, label, score in detections:
                class_aggregates[label] = class_aggregates.get(label, 0) + 1

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
                    if fps > 0:
                        fps_samples.append(fps)

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
            logger.info(f"Exported {len(tracker.exit_stats)} object statistics (CSV)")
            
            import uuid
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            run_id = str(uuid.uuid4())[:8]
            
            export_run_json(
                run_id=run_id,
                start_time=start_time_str,
                end_time=end_time_str,
                duration_seconds=end_time - start_time,
                model=model,
                device="cuda" if hasattr(model, "device") and "cuda" in str(model.device) else "cpu",
                confidence_threshold=conf,
                input_type="video" if is_video else "camera",
                resolution=resolution or (640, 480),
                total_frames=frames,
                fps_samples=fps_samples,
                objects=objects_dict,
                class_aggregates=class_aggregates,
                output_dir=output_dir,
            )
            logger.info(f"Exported run metadata (JSON) - Run ID: {run_id}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

        if benchmark:
            duration = end_time - start_time
            fps = frames / duration if duration > 0 else 0
            logger.info(f"Benchmark - Frames: {frames}, Duration: {duration:.2f}s, Avg FPS: {fps:.2f}")


