from pathlib import Path
from typing import Dict, Any, Optional, Set

class ValidationError(Exception):
    pass

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValidationError("Config must be dict")
    
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValidationError("model must be dict")
    
    model_path = model_cfg.get("path", "models/yolov8n.pt")
    device = model_cfg.get("device", "cpu").lower()
    
    if device not in ("cpu", "cuda", "mps"):
        raise ValidationError(f"device must be cpu/cuda/mps, got {device}")
    
    model_cfg["path"] = model_path
    model_cfg["device"] = device
    
    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise ValidationError("runtime must be dict")
    
    mode = runtime_cfg.get("mode", "live").lower()
    if mode not in ("live", "video", "headless", "benchmark"):
        raise ValidationError(f"mode must be live/video/headless/benchmark, got {mode}")
    
    confidence = runtime_cfg.get("confidence", 0.5)
    if not (0 <= confidence <= 1):
        raise ValidationError(f"confidence must be 0-1, got {confidence}")
    
    camera_index = runtime_cfg.get("camera_index", 0)
    if not isinstance(camera_index, int) or camera_index < 0:
        raise ValidationError(f"camera_index must be non-negative int, got {camera_index}")
    
    max_fps = runtime_cfg.get("max_fps")
    if max_fps is not None and max_fps <= 0:
        raise ValidationError(f"max_fps must be positive, got {max_fps}")
    
    video_path = runtime_cfg.get("video_path")
    if mode == "video" and not video_path:
        raise ValidationError("video_path required for video mode")
    
    save_video = runtime_cfg.get("save_video", False)
    if not isinstance(save_video, bool):
        raise ValidationError(f"save_video must be bool, got {save_video}")
    
    runtime_cfg.update({
        "mode": mode,
        "confidence": confidence,
        "camera_index": camera_index,
        "max_fps": max_fps,
        "video_path": video_path,
        "save_video": save_video,
    })
    
    config["model"] = model_cfg
    config["runtime"] = runtime_cfg
    
    return config


def validate_model_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        if not p.is_file():
            raise ValidationError(f"Model path not a file: {path}")
        return p
    if p.suffix.lower() == ".pt" or path.startswith("yolov8"):
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p
    raise ValidationError(f"Model not found: {path}")


def validate_video_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"Video not found: {path}")
    if not p.is_file():
        raise ValidationError(f"Video path not a file: {path}")
    valid = {".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    if p.suffix.lower() not in valid:
        raise ValidationError(f"Unsupported video format: {p.suffix}")
    return p

def validate_frame(frame) -> bool:
    if frame is None:
        raise ValidationError("Frame is None")
    if len(frame.shape) != 3:
        raise ValidationError(f"Frame must be 3D, got shape {frame.shape}")
    if frame.size == 0:
        raise ValidationError("Frame is empty")
    return True
