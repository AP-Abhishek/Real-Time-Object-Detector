# Real-Time Object Detection Using YOLOv8

This project implements a real-time object detection system using a webcam. It uses the YOLOv8 model to identify and label objects such as people and everyday items by drawing bounding boxes on live video frames.

### Features

- Real-time object detection using YOLOv8
- Webcam input support
- Video file input support
- Bounding boxes and class labels rendered in real time
- Confidence score displayed per detection
- Real-time FPS display
- Configurable confidence threshold via CLI
- Dynamic class filtering via CLI
- Optional FPS limiting
- Centroid-based object tracking
- Stable object IDs
- Detection-to-tracking association
- Object enter and exit event detection
- Structured logging
- Per-object lifetime tracking
- Per-object frame count tracking
- Lifetime displayed in bounding box overlay
- Active object count overlay
- Lifetime and frame statistics logged on object exit
- Clean and deterministic shutdown

### Tools Used
- Python
- OpenCV
- Ultralytics YOLOv8
- uv (Python package manager)

### Folder Structure
```
real-time-object-detector/
│
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── config.py
│   ├── detector.py
│   └── logger.py
│
├── models/
│   └── yolov8n.pt
│
├── requirements.txt
├── README.md
├── main.py
├── .python-version
├── pyproject.toml
└── .gitignore
```