import csv
import os
import time

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
