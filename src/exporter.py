import csv
import os
import time
import json

def export_exit_stats_csv(exit_stats, output_dir="exports"):
    if not exit_stats:
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"run_{timestamp}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["object_id", "lifetime_seconds", "total_frames", "exit_timestamp"]
        )
        for object_id, (lifetime, frames) in exit_stats.items():
            writer.writerow(
                [
                    object_id,
                    round(lifetime, 2),
                    frames,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

def export_run_json(
    run_id,
    start_time,
    end_time,
    duration_seconds,
    model,
    device,
    confidence_threshold,
    input_type,
    resolution,
    total_frames,
    fps_samples,
    objects,
    class_aggregates,
    output_dir="exports",
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"run_{timestamp}.json")

    payload = {
        "run_metadata": {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": round(duration_seconds, 2),
            "model": {
                "name": getattr(model, "model_name", None),
                "version": None,
                "path": None,
            },
            "device": device,
            "confidence_threshold": confidence_threshold,
            "input_source": {
                "type": input_type,
                "value": None,
            },
            "resolution": {
                "width": resolution[0],
                "height": resolution[1],
            },
            "performance": {
                "total_frames": total_frames,
                "avg_fps": sum(fps_samples) / len(fps_samples) if fps_samples else 0.0,
                "max_fps": max(fps_samples) if fps_samples else 0.0,
            },
        },
        "objects": objects,
        "class_aggregates": class_aggregates,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
