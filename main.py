import cv2
from src.camera import open_camera, read_frame, release_camera
from src.detector import load_model, detect

def main():
    cap = open_camera(0)
    model = load_model()

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        annotated_frame = detect(model, frame)
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(cap)

if __name__ == "__main__":
    main()
