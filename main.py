import cv2
import time
from src.camera import open_camera, read_frame, release_camera
from src.detector import load_model, detect

def main():
    cap = open_camera(0)
    model = load_model("models/yolov8n.pt")

    conf_threshold = 0.5
    allowed_classes = ["person"]

    prev_time = 0.0

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        detections = detect(model, frame, conf_threshold, allowed_classes)
        
        for x1, y1, x2, y2, label, conf in detections:
            text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(cap)

if __name__ == "__main__":
    main()
