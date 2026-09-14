# Real-Time Object Detection - Computer Vision & YOLOv8 Inference Pipeline

An intelligent computer vision application built with **Python**, **OpenCV**, **Ultralytics YOLOv8**, **SciPy**, and **PyYAML**. It enables real-time object detection and multi-object centroid tracking across live webcam feeds and video files with full analytics reporting (CSV/JSON), optional annotated video recording, class filtering, and high-performance execution modes (`live`, `video`, `headless`, `benchmark`).

---

## Project Overview & Application Architecture

The **Real-Time Object Detection** system leverages an end-to-end vision pipeline to process video frames, identify multi-class targets using neural networks, track object movement across frames, and export structured operational analytics.

The core application architecture follows a modular vision & tracking pipeline:

1. **Configuration & Validation (`src/config_loader.py` & `src/validation.py`):** Loads and validates YAML settings (`config.yaml`) and CLI parameters, validating confidence thresholds, camera indices, frame rates, image sizes (`imgsz`), model paths, and device parameters (`cpu`, `cuda`, `mps`).
2. **Device Detection & Fallback (`src/device_utils.py`):** Checks GPU availability (CUDA / Apple Silicon MPS) and automatically selects the optimal hardware compute device with automatic CPU fallback.
3. **Model Loader & Downloader (`src/model.py`):** Initializes the Ultralytics YOLOv8 architecture (e.g. `yolov8n.pt`, `yolov8s.pt`), automatically fetching standard pretrained weights if missing locally.
4. **Video Stream Handling (`src/camera.py`):** Wraps OpenCV `VideoCapture` streams for low-latency frame reading from webcams or video files with graceful stream termination.
5. **Detector Engine & Class Filtering (`src/detector.py`):** Runs YOLOv8 inference per frame, converts bounding box structures, applies confidence thresholds, and filters detections against allowed COCO target classes.
6. **Centroid Multi-Object Tracker (`src/tracker.py`):** Class-aware and box-aware `CentroidTracker` using SciPy Euclidean distance matrices to track object movement, assign unique IDs, record lifetime durations, and detect frame entry/exit events.
7. **Runtime Analytics & Serialized Exporter (`src/runtime_tracker.py` & `src/exporter.py`):** Aggregates frame rates, object counts, per-class lifetime statistics, and exports structured reports (`summary.json`, `summary.csv`, `run_index.json`).
8. **Pipeline Engine & GUI Overlay (`src/pipeline.py` & `main.py`):** Drives the processing loop, renders bounding boxes and performance overlays, records output video (`VideoWriter`), and provides the `main.py` entry point with custom formatted help menus.

---

## Core Modules & Features

* **Real-Time Object Detection:** Powered by Ultralytics YOLOv8 for precise multi-class object detection across live video feeds.
* **Class-Aware Centroid Tracking:** Stable multi-object tracking with persistent object IDs, entrance/exit event triggers, and lifetime duration monitoring.
* **Dual Input Sources:** Supports live webcam hardware feeds as well as video file inputs (`--video path/to/video.mp4`).
* **Flexible Execution Modes:** Four distinct runtime operating modes:
  * `live`: Interactive GUI window displaying real-time webcam feed with object bounding boxes.
  * `video`: Processes input video files with on-screen bounding boxes and FPS overlays.
  * `headless`: Background processing without rendering GUI display windows (ideal for headless servers).
  * `benchmark`: Automated performance benchmark mode calculating total frames, elapsed time, and average FPS.
* **Annotated Video Saving (`--save-video`):** Optionally records fully annotated detection feeds to MP4 video files using OpenCV `VideoWriter`.
* **Configurable Class Filtering:** Filter detections to specific target COCO classes (e.g., `person`, `car`, `dog`) via CLI or configuration.
* **Automatic Hardware Acceleration:** Auto-detects CUDA GPUs or Apple Silicon MPS, gracefully falling back to CPU when acceleration is unavailable.
* **Comprehensive Analytics Export:** Automatically generates per-run structured reports (`summary.csv`, `summary.json`, `run_index.json`) containing object statistics, total frames, and lifetime aggregates.

---

## Data & Pipeline Details

The system processes video feeds through structured pipeline modules:

| Component / Module | Implementation File | Description & Functionality |
| :--- | :--- | :--- |
| **Config Loader** | `src/config_loader.py` | Parses YAML configurations and merges command-line parameter overrides. |
| **Validator Engine** | `src/validation.py` | Validates model paths, confidence ranges, device selections, FPS limits, and `imgsz`. |
| **Device Utility** | `src/device_utils.py` | Detects CUDA/MPS hardware capabilities and enforces CPU fallback when needed. |
| **Model Loader** | `src/model.py` | Loads YOLO neural network instances and auto-downloads standard pretrained weights. |
| **Camera Interface** | `src/camera.py` | Opens, reads frames, and safely releases webcam hardware or video file streams. |
| **Detection Engine** | `src/detector.py` | Performs YOLOv8 object inference, confidence thresholding, and target class filtering. |
| **Centroid Tracker** | `src/tracker.py` | Associates bounding box centroids across frames using Euclidean distance matrix calculations. |
| **Runtime Tracker** | `src/runtime_tracker.py` | Tracks object entrance/exit events, active frame counts, and per-class aggregate metrics. |
| **Results Exporter** | `src/exporter.py` | Serializes session analytics to `summary.csv`, `summary.json`, and `run_index.json`. |
| **Logger Module** | `src/logger.py` | Configures structured logging to console streams and log files. |
| **Pipeline Runner** | `src/pipeline.py` | Orchestrates the main execution loop, rendering, video recording, and resource teardown. |
| **Entry Point** | `main.py` | Development entry point featuring custom help formatting, signal handling, and argument parsing. |

---

## Tech Stack & Specifications

* **Language:** Python `>=3.11`
* **Computer Vision:** OpenCV (`opencv-python>=4.13.0`)
* **Deep Learning & Inference:** Ultralytics YOLOv8 (`ultralytics>=8.4.8`), PyTorch
* **Mathematical Utilities:** SciPy (`scipy>=1.17.0`), NumPy (`numpy>=2.4.1`)
* **Configuration & Serialization:** PyYAML (`pyyaml>=6.0.3`)
* **Packaging & Tools:** Setuptools (`setuptools>=61.0`)
* **Testing Suite:** Pytest (`pytest>=9.0.2`)

---

## Environment Setup & Local Development

Follow these steps to set up the development environment and run the application locally:

### Prerequisites
* **Python 3.11** or higher installed.
* **Git** installed.
* **Webcam** (optional, required for live mode).

### Step 1: Clone the Repository
```bash
git clone https://github.com/AP-Abhishek/Real-Time-Object-Detector.git
cd Real-Time-Object-Detector
```

### Step 2: Create & Activate Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Development Script (`main.py`)

#### 1. Live Webcam Detection (Default)
```bash
python main.py
```

#### 2. Video File Detection
```bash
python main.py --video path/to/video.mp4
```

#### 3. Save Annotated Video Output
```bash
python main.py --save-video --output runs/detection_run
```

#### 4. Headless & Benchmark Modes
```bash
# Run without rendering display window
python main.py --headless

# Performance benchmark mode
python main.py --benchmark
```

#### 5. Custom Model & Confidence Threshold
```bash
python main.py --model models/yolov8s.pt --confidence 0.7 --device cpu
```

### Command Line Arguments Reference

```text
usage: python main.py [-h] [--config CONFIG] [--mode {live,video,headless,benchmark}] [--confidence [0-1]]
                      [--camera INDEX] [--video PATH] [--output DIR] [--model PATH] [--device {cpu,cuda,mps}]
                      [--max-fps N] [--headless] [--benchmark] [--save-video] [-v]

Real-Time Object Detection using YOLOv8

options:
  -h, --help                              show this help message and exit
  --config CONFIG                         Config file path (default: config.yaml)
  --mode {live,video,headless,benchmark}  Execution mode
  --confidence [0-1]                      Detection confidence threshold
  --camera INDEX                          Camera device index
  --video PATH                            Video file path (sets mode to video)
  --output DIR                            Output directory
  --model PATH                            Model file path
  --device {cpu,cuda,mps}                 Compute device
  --max-fps N                             Maximum FPS limit
  --headless                              No display window (headless mode)
  --benchmark                             Benchmark mode (no display, print metrics)
  --save-video                            Save annotated video output to file
  -v, --version                           show program's version number and exit
```

### Step 5: Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```text
Real-Time-Object-Detector/
├── models/                            # YOLOv8 model weights (.pt files)
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── camera.py                      # OpenCV VideoCapture wrappers for camera and video
│   ├── config.py                      # Config dataclass and schema definitions
│   ├── config_loader.py               # YAML config loader and validator integration
│   ├── detector.py                    # YOLOv8 inference engine and class filtering
│   ├── device_utils.py                # CUDA / MPS hardware detection and CPU fallback
│   ├── exporter.py                    # Export summary JSON, CSV, and index files
│   ├── logger.py                      # Logging setup and stream/file handler configuration
│   ├── model.py                       # Ultralytics model loader and auto-downloader
│   ├── pipeline.py                    # Main detection pipeline, rendering, and recording
│   ├── runtime_tracker.py             # Per-session object lifecycle and class aggregate tracker
│   ├── tracker.py                     # CentroidTracker for multi-object tracking and distance association
│   └── validation.py                  # Input validation rules and ValidationError exception
├── tests/                             # Automated test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures and mock objects
│   ├── test_config_loader.py          # Config loading and schema validation tests
│   ├── test_detector.py              # YOLO inference wrapper tests
│   ├── test_exporter.py              # CSV/JSON summary export tests
│   ├── test_logger.py                # Logging output tests
│   ├── test_tracker.py               # Centroid tracker association tests
│   └── test_validation.py            # Input parameter validation tests
├── config.yaml                        # Default system configuration file
├── main.py                            # Development entry point and argument parser
├── pyproject.toml                     # Project dependencies and metadata manifest
├── requirements.txt                   # Dependency requirements list
├── verify_setup.py                    # Setup verification utility script
└── README.md                          # Comprehensive project documentation
```

---

## Copyright

© 2026 **AP-Abhishek**. All rights reserved.
