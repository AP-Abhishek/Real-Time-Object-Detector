# Real-Time Object Detection Using YOLOv8

This project implements a real-time object detection system using a webcam. It uses the YOLOv8 model to identify and label objects such as people and everyday items by drawing bounding boxes on live video frames.

### Features

- Real-time object detection (YOLOv8 neural network)
- Webcam and video file input
- Live FPS tracking and display
- Stable object tracking with IDs (centroid-based)
- Per-object lifecycle tracking (lifetime, frame count)
- CSV and JSON export of detection results
- 4 operating modes: live, video, headless, benchmark
- GPU support with automatic CPU fallback
- Configurable confidence threshold and class filtering
- FPS limiting for resource control
- Structured logging (console and file)
- Full error handling and input validation
- Type hints throughout codebase
- CLI arguments for runtime configuration

### Tools Used
- Python 3.11+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- SciPy
- PyYAML
- uv (Python package manager)

### Installation

#### Prerequisites
- Python 3.11 or higher
- pip or uv package manager
- Webcam (for live mode)

#### Quick Setup

```bash
git clone https://github.com/AP-Abhishek/Real-Time-Object-Detector.git
cd real-time-object-detection

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv add -r requirements.txt
```

### Quick Start

```bash
python main.py
```

Press 'q' or ESC to exit.

### Configuration

Edit `config.yaml`:

```yaml
model:
  path: models/yolov8n.pt
  device: cpu

runtime:
  mode: live                    # live, video, headless, benchmark
  confidence: 0.5
  camera_index: 0
  output_dir: runs/latest
  max_fps: null                 # Unlimited
  video_path: null
  allowed_classes: null
```

### Usage Modes

**Live**: Default (webcam)
**Video**: Set `mode: video` and `video_path: path/to/video.mp4`
**Headless**: Set `mode: headless` (no display)
**Benchmark**: Set `mode: benchmark` (performance metrics)

### Folder Structure

```
real-time-object-detection/
│
├── src/
│   ├── __init__.py
│   ├── camera.py              # Camera/video input
│   ├── config_loader.py       # Config loading
│   ├── config.py              # Config class (legacy)
│   ├── detector.py            # YOLOv8 inference
│   ├── exporter.py            # Results export
│   ├── logger.py              # Logging
│   ├── model.py               # Model loading
│   ├── pipeline.py            # Main pipeline
│   ├── runtime_tracker.py     # Runtime tracking
│   ├── tracker.py             # Object tracking
│   ├── validation.py          # Input validation
│   └── __pycache__/
│
├── models/
│   └── yolov8n.pt
│
├── exports/
├── runs/
├── config.yaml
├── main.py
├── requirements.txt
├── pyproject.toml
├── .python-version
├── .gitignore
└── README.md
```

### Output

Results saved to `output_dir`:
- `summary.csv` - Object stats
- `summary.json` - Run metadata
- Logs - Execution details

### Error Handling

Gracefully handles:
- Missing camera/video
- Invalid config
- Model load failures
- Frame errors
- Resource cleanup

### Performance Tips

- GPU: Set `device: cuda`
- Faster: Use `yolov8n.pt`
- Lower load: Set `max_fps: 15`

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera won't open | Check permissions, try different index |
| Slow | Use nano model or enable GPU |
| Model not found | Ensure `models/yolov8n.pt` exists |
| Config errors | Validate YAML syntax |

### Development

Add features:
1. Implement in module
2. Add type hints
3. Add validation
4. Add logging
5. Test

### References

- YOLOv8: https://docs.ultralytics.com/
- OpenCV: https://opencv.org/
- PyTorch: https://pytorch.org/
