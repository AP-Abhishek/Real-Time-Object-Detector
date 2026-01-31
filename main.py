import cv2
import time
import torch

from src.camera import open_camera, read_frame, release_camera
from src.detector import load_model, detect
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, CAMERA_INDEX, WINDOW_NAME

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cap = open_camera(CAMERA_INDEX)
    model = load_model(MODEL_PATH, device)

    prev_time = 0.0

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        detections = detect(model, frame, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES)
        
        for x1, y1, x2, y2, label, conf in detections:
            text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Detections", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

if __name__ == "__main__":
    main()
