import torch
from typing import Literal

def detect_available_device() -> Literal["cpu", "cuda", "mps"]:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_device_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    return info

def validate_and_fallback(device: str) -> str:
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return "cpu"
