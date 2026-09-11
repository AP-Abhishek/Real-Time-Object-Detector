import numpy as np
import time
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared: int = 30) -> None:
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

        self.boxes = {}
        self.labels = {}
        self.scores = {}

        self.start_time = {}
        self.total_frames = {}
        self.exit_stats = {}
        
        self.entered = set()
        self.exited = set()

    def register(
        self,
        centroid: np.ndarray,
        box: Tuple[int, int, int, int] = (0, 0, 0, 0),
        label: str = "object",
        score: float = 0.0,
    ) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.boxes[self.next_object_id] = box
        self.labels[self.next_object_id] = label
        self.scores[self.next_object_id] = score

        self.start_time[self.next_object_id] = time.time()
        self.total_frames[self.next_object_id] = 1


        self.entered.add(self.next_object_id)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        lifetime = time.time() - self.start_time[object_id]
        frames = self.total_frames[object_id]

        self.exited.add(object_id)
        self.exit_stats[object_id] = (lifetime, frames)

        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.boxes:
            del self.boxes[object_id]
        if object_id in self.labels:
            del self.labels[object_id]
        if object_id in self.scores:
            del self.scores[object_id]
        del self.start_time[object_id]
        del self.total_frames[object_id]

    def deregister_all(self) -> None:
        for object_id in list(self.objects.keys()):
            self.deregister(object_id)

    def get_tracked_info(self) -> Dict[int, Dict[str, Any]]:
        info = {}
        for oid in self.objects:
            info[oid] = {
                "centroid": self.objects[oid],
                "box": self.boxes.get(oid, (0, 0, 0, 0)),
                "label": self.labels.get(oid, "object"),
                "score": self.scores.get(oid, 0.0),
                "start_time": self.start_time.get(oid, time.time()),
                "total_frames": self.total_frames.get(oid, 0),
            }
        return info

    def update(self, rects: List[Any]) -> Tuple[OrderedDict, Dict[str, Any]]:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            events = {
                "entered": list(self.entered),
                "exited": list(self.exited),
            }

            self.entered.clear()
            self.exited.clear()

            return self.objects, events

        boxes_in = []
        labels_in = []
        scores_in = []

        for item in rects:
            x1, y1, x2, y2 = item[0], item[1], item[2], item[3]
            label = item[4] if len(item) > 4 else "object"
            score = item[5] if len(item) > 5 else 0.0
            boxes_in.append((x1, y1, x2, y2))
            labels_in.append(label)
            scores_in.append(score)

        input_centroids = np.zeros((len(boxes_in), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(boxes_in):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(
                    input_centroids[i],
                    box=boxes_in[i],
                    label=labels_in[i],
                    score=scores_in[i],
                )
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = boxes_in[col]
                self.labels[object_id] = labels_in[col]
                self.scores[object_id] = scores_in[col]
                self.disappeared[object_id] = 0
                self.total_frames[object_id] += 1
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(
                        input_centroids[col],
                        box=boxes_in[col],
                        label=labels_in[col],
                        score=scores_in[col],
                    )

        events = {
            "entered": list(self.entered),
            "exited": list(self.exited),
        }

        self.entered.clear()
        self.exited.clear()

        return self.objects, events

