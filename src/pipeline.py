import time
import cv2
import uuid
import os
from collections import defaultdict

from src.camera import read_frame, release_camera
from src.detector import detect
from src.tracker import CentroidTracker
from src.config import WINDOW_NAME
from src.exporter import export_exit_stats_csv, export_run_json

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

    run_id = str(uuid.uuid4())
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)

    start_ts = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start_ts))

    frame_interval = 1.0 / max_fps if max_fps and max_fps > 0 else None
    prev_time = None
    last_frame_time = 0.0
    running = True

    total_frames = 0
    fps_samples = []

    object_classes = {}
    object_first_seen = {}
    object_last_seen = {}

    class_aggregates = defaultdict(lambda: {
        "class_id": None,
        "object_count": 0,
        "total_time_seconds": 0.0,
        "max_time_seconds": 0.0,
        "total_frames": 0,
        "avg_time_seconds": 0.0
    })

    while running:
        now = time.time()
        if frame_interval and (now - last_frame_time) < frame_interval:
            time.sleep(frame_interval - (now - last_frame_time))
        last_frame_time = time.time()

        ret, frame = cap.read() if is_video else read_frame(cap)
        if not ret or frame is None:
            break

        total_frames += 1

        detections = detect(model, frame, conf, allowed_classes)
        rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections]
        objects, events = tracker.update(rects)

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

        for oid in events["entered"]:
            object_first_seen[oid] = now_iso

        for oid in events["exited"]:
            object_last_seen[oid] = now_iso

        for (x1, y1, x2, y2, label, det_conf) in detections:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)

            object_id = None
            min_dist = float("inf")

            for oid, (oX, oY) in objects.items():
                d = (cX - oX) ** 2 + (cY - oY) ** 2
                if d < min_dist:
                    min_dist = d
                    object_id = oid

            if object_id not in object_classes:
                object_classes[object_id] = label
                class_aggregates[label]["object_count"] += 1

            lifetime = tracker.exit_stats.get(object_id, (0, 0))[0]
            text = f"ID {object_id} | {label} {det_conf:.2f} | {int(lifetime)}s"

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
        if prev_time:
            fps = 1 / (current_time - prev_time)
            if fps > 0:
                fps_samples.append(fps)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps_samples[-1]:.2f}" if fps_samples else "FPS: --",
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

    end_ts = time.time()
    end_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(end_ts))
    duration = end_ts - start_ts

    for oid in tracker.objects.keys():
        if oid not in object_last_seen:
            object_last_seen[oid] = end_iso

    objects_json = []

    for oid, (lifetime, frames) in tracker.exit_stats.items():
        cls = object_classes.get(oid)

        objects_json.append({
            "object_id": oid,
            "class_id": None,
            "class_name": cls,
            "frames_seen": frames,
            "time_visible_seconds": round(lifetime, 2),
            "first_seen": object_first_seen.get(oid),
            "last_seen": object_last_seen.get(oid)
        })

        agg = class_aggregates[cls]
        agg["total_time_seconds"] += lifetime
        agg["total_frames"] += frames
        agg["max_time_seconds"] = max(agg["max_time_seconds"], lifetime)

    for agg in class_aggregates.values():
        if agg["object_count"] > 0:
            agg["avg_time_seconds"] = agg["total_time_seconds"] / agg["object_count"]

    release_camera(cap)
    cv2.destroyAllWindows()

    export_exit_stats_csv(tracker.exit_stats, run_dir)

    export_run_json(
        run_id=run_id,
        start_time=start_iso,
        end_time=end_iso,
        duration_seconds=duration,
        model=model,
        device=str(model.device),
        confidence_threshold=conf,
        input_type="video" if is_video else "webcam",
        resolution=(frame.shape[1], frame.shape[0]),
        total_frames=total_frames,
        fps_samples=fps_samples,
        objects=objects_json,
        class_aggregates=class_aggregates,
        output_dir=run_dir,
    )

    logger.info("Application terminated gracefully.")
