import time
import cv2

from src.camera import read_frame, release_camera
from src.detector import detect
from src.tracker import CentroidTracker
from src.config import WINDOW_NAME


def run_pipeline(
    cap,
    model,
    logger,
    conf,
    allowed_classes,
    max_fps=None,
    is_video=False,
):
    tracker = CentroidTracker()

    frame_interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
    prev_time = 0.0
    last_frame_time = 0.0
    running = True

    while running:
        now = time.time()
        if frame_interval and (now - last_frame_time) < frame_interval:
            time.sleep(frame_interval - (now - last_frame_time))
        last_frame_time = time.time()

        ret, frame = cap.read() if is_video else read_frame(cap)
        if not ret or frame is None:
            logger.warning("Input stream ended.")
            break

        detections = detect(model, frame, conf, allowed_classes)

        rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections]
        objects, events = tracker.update(rects)

        for object_id in events["entered"]:
            logger.info(f"Object {object_id} entered frame")

        for object_id in events["exited"]:
            logger.info(f"Object {object_id} exited frame")


        for (x1, y1, x2, y2, label, conf) in detections:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            object_id = None
            min_dist = float("inf")

            for oid, (oX, oY) in objects.items():
                d = (cX - oX) ** 2 + (cY - oY) ** 2
                if d < min_dist:
                    min_dist = d
                    object_id = oid

            text = f"ID {object_id}: {label} {conf:.2f}" if object_id is not None else f"{label} {conf:.2f}"

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

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            running = False

    release_camera(cap)
    cv2.destroyAllWindows()
    logger.info("Application terminated gracefully.")
