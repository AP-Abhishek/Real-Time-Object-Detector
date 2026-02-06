import cv2
from src.validation import ValidationError, validate_video_path

def open_camera(index=0):
    if not isinstance(index, int) or index < 0:
        raise ValidationError(f"Camera index must be non-negative int")
    
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at index {index}")
    
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Camera opened but failed to read frames")
    
    return cap

def open_video(path):
    validate_video_path(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Video opened but contains no readable frames: {path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap

def read_frame(cap):
    try:
        ret, frame = cap.read()
        return ret, frame
    except Exception as e:
        raise RuntimeError(f"Error reading frame: {e}")

def release_camera(cap):
    if cap is not None:
        try:
            cap.release()
        except Exception as e:
            print(f"Warning: Error releasing camera: {e}")
