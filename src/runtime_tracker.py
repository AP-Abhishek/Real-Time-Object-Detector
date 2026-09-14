import os
import time
import uuid
import json
from collections import defaultdict
from src.exporter import export_exit_stats_csv, export_run_json

from src import __version__ as VERSION

class RuntimeTracker:
    def __init__(
        self,
        model,
        device,
        confidence_threshold,
        input_type,
        resolution,
        output_root,
    ):
        self.run_id = str(uuid.uuid4())[:8]
        self.version = VERSION

        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.input_type = input_type
        self.resolution = resolution

        self.run_dir = output_root
        os.makedirs(self.run_dir, exist_ok=True)

        self.start_time = time.time()
        self.end_time = None

        self.object_start = {}
        self.object_frames = defaultdict(int)
        self.exit_stats = {}

        self.objects = {}

        self.class_aggregates = defaultdict(lambda: {
            "count": 0,
            "total_lifetime": 0.0,
            "total_frames": 0,
            "avg_lifetime": 0.0,
            "avg_frames": 0.0
        })

        self.total_frames = 0
        self.fps_samples = []

    def tick_frame(self, fps):
        self.total_frames += 1
        self.fps_samples.append(fps)

    def register_object(self, object_id, class_name):
        if object_id in self.object_start:
            return

        now = time.time()

        self.object_start[object_id] = now
        self.object_frames[object_id] = 0

        self.objects[object_id] = {
            "class": class_name,
            "first_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "last_seen": None,
            "lifetime_seconds": 0.0,
            "total_frames": 0,
        }

        self.class_aggregates[class_name]["count"] += 1

    def update_object(self, object_id):
        if object_id in self.object_frames:
            self.object_frames[object_id] += 1

    def unregister_object(self, object_id):
        if object_id not in self.object_start:
            return

        now = time.time()
        lifetime = now - self.object_start[object_id]
        frames = self.object_frames[object_id]

        self.exit_stats[object_id] = (lifetime, frames)

        obj = self.objects[object_id]
        obj["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        obj["lifetime_seconds"] = round(lifetime, 2)
        obj["total_frames"] = frames

        agg = self.class_aggregates[obj["class"]]
        agg["total_lifetime"] += lifetime
        agg["total_frames"] += frames

        del self.object_start[object_id]
        del self.object_frames[object_id]

    def unregister_all(self):
        for object_id in list(self.object_start.keys()):
            self.unregister_object(object_id)

    def _finalize_class_aggregates(self):
        for data in self.class_aggregates.values():
            if data["count"] > 0:
                data["avg_lifetime"] = round(
                    data["total_lifetime"] / data["count"], 2
                )
                data["avg_frames"] = int(
                    data["total_frames"] / data["count"]
                )

    def _export_run_index(self):
        index_path = os.path.join(self.run_dir, "run_index.json")

        payload = {
            "run_id": self.run_id,
            "version": self.version,
            "start_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)
            ),
            "end_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)
            ),
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "total_frames": self.total_frames,
            "avg_fps": sum(self.fps_samples) / len(self.fps_samples)
            if self.fps_samples
            else 0.0,
        }

        with open(index_path, "w") as f:
            json.dump(payload, f, indent=2)

    def finalize(self):
        self.end_time = time.time()
        self._finalize_class_aggregates()

        export_exit_stats_csv(self.exit_stats, self.run_dir)

        export_run_json(
            run_id=self.run_id,
            start_time=time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)
            ),
            end_time=time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)
            ),
            duration_seconds=self.end_time - self.start_time,
            model=self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            input_type=self.input_type,
            resolution=self.resolution,
            total_frames=self.total_frames,
            fps_samples=self.fps_samples,
            objects=self.objects,
            class_aggregates=self.class_aggregates,
            output_dir=self.run_dir,
        )

        self._export_run_index()
